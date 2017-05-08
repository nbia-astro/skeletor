from skeletor import Float, Float3, Field, Particles
from skeletor import Ohm, Faraday, State
from skeletor.manifolds.second_order import Manifold
from skeletor.time_steppers.predictor_corrector import TimeStepper
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from dispersion_solvers import HallDispersion
from scipy.special import erfinv

plot = False
fitplot = True
# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 32, 1
# Grid size in x- and y-direction (square cells!)
Lx = nx
Ly = Lx*ny/nx
# Average number of particles per cell
npc = 2**12
# Particle charge and mass
charge = 0.19634954084936207
mass = 1.0
# Electron temperature
Te = 0.0
# Dimensionless amplitude of perturbation
A = 0.005
# Wavenumbers
ikx = 1
iky = 0
# CFL number
cfl = 0.8
# Number of periods to run for
nperiods = 2

# Sound speed
cs = np.sqrt(Te/mass)

# Total number of particles in simulation
N = npc*nx*ny

# Wave vector and its modulus
kx = 2*np.pi*ikx/Lx
ky = 2*np.pi*iky/Ly
k = np.sqrt(kx*kx + ky*ky)

# Angle of k-vector with respect to x-axis
theta = np.arctan(iky/ikx) if ikx != 0 else np.pi/2

# Magnetic field strength
B0 = 1

(Bx, By, Bz) = (B0*np.cos(theta), B0*np.sin(theta), 0)
B2 = Bx**2 + By**2 + Bz**2

rho0 = 1.0

va = B0

# Ohmic resistivity
eta = 0

beta = 20

vt = np.sqrt(va**2*beta/2)
vtx, vty, vtz = vt, vt, vt

# Cyclotron frequency
oc = charge*B0/mass

# Hall parameter
etaH = va**2/oc

di = HallDispersion(kperp=0, kpar=k, va=va, cs=cs, etaH=etaH, eta=eta,
                    along_x=True, theta=theta)

# Mode number
m = 0

omega = -0.14922889208545562+0.095051744736327104j

def frequency(kzva):  # noqa: E302
    hel = 1
    return kzva*(np.sqrt(1.0 + (0.5*kzva/oc)**2) + 0.5*kzva/(hel*oc))

def get_dt(kzva):  # noqa: E302
    dt = cfl/frequency(kzva)
    dt = 2.0**(np.floor(np.log2(dt)))
    return dt


kmax = np.pi

vph = omega.real/kmax

# Simulation time
tend = 20.

# Phase factor
def phase(x, y, t):  # noqa: E302
    return A*np.exp(1j*(omega*t - kx*x - ky*y))

# Linear solutions in real space
def Bx_an(x, y, t):  # noqa: E302
    return Bx + B0*(di.vec[m]['bx']*phase(x, y, t))
def By_an(x, y, t):  # noqa: E302
    return By + B0*(di.vec[m]['by']*phase(x, y, t))
def Bz_an(x, y, t):  # noqa: E302
    return Bz + B0*(di.vec[m]['bz']*phase(x, y, t))
def Ux_an(x, y, t):  # noqa: E302
    return (di.vec[m]['vx']*phase(x, y, t))
def Uy_an(x, y, t):  # noqa: E302
    return (di.vec[m]['vy']*phase(x, y, t))
def Uz_an(x, y, t):  # noqa: E302
    return (di.vec[m]['vz']*phase(x, y, t))


# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(np.sqrt(npc))
assert sqrt_npc**2 == npc
a = (np.arange(sqrt_npc) + 0.5)/sqrt_npc
x_cell, y_cell = np.meshgrid(a, a)
x_cell = x_cell.flatten()
y_cell = y_cell.flatten()

R = (np.arange(npc) + 0.5)/npc
vx_cell = erfinv(2*R - 1)*np.sqrt(2)*vtx
vy_cell = erfinv(2*R - 1)*np.sqrt(2)*vty
vz_cell = erfinv(2*R - 1)*np.sqrt(2)*vtz
np.random.shuffle(vx_cell)
np.random.shuffle(vy_cell)
np.random.shuffle(vz_cell)
for i in range(nx):
    for j in range(ny):
        if i == 0 and j == 0:
            x = x_cell + i
            y = y_cell + j
            vx = vx_cell
            vy = vy_cell
            vz = vz_cell
        else:
            x = np.concatenate((x, x_cell + i))
            y = np.concatenate((y, y_cell + j))
            vx = np.concatenate((vx, vx_cell))
            vy = np.concatenate((vy, vy_cell))
            vz = np.concatenate((vz, vz_cell))

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=nx, Ly=ny)

# Time step
# dt = cfl*manifold.dx/vph
dt = get_dt(kmax*va)

# Number of time steps
nt = int(tend/dt)

faraday = Faraday(manifold)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Pair of Fourier basis functions with the specified wave numbers.
# The basis functions are normalized so that the Fourier amplitude can be
# computed by summing rather than averaging.
S = np.sin(kx*xg + ky*yg)/(nx*ny)
C = np.cos(kx*xg + ky*yg)/(nx*ny)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, vz)

# Create a uniform density field
# init = InitialCondition(npc, quiet=quiet, vt=vt)
# init(manifold, ions)

# Perturbation to particle velocities
ions['vx'] += Ux_an(ions['x'], ions['y'], t=-dt/2).real
ions['vy'] += Uy_an(ions['x'], ions['y'], t=-dt/2).real
ions['vz'] += Uz_an(ions['x'], ions['y'], t=-dt/2).real

def B_an(t):  # noqa: E302
    B_an = Field(manifold, dtype=Float3)
    B_an['x'].active = Bx_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t).real
    B_an['y'].active = By_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t).real
    B_an['z'].active = Bz_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t).real
    return B_an


# Create vector potential
A_an = Field(manifold, dtype=Float3)
A_an['x'].active = 0
A_an['y'].active = -(Bz_an(xg, yg, -dt/2)/(1j*kx)).real
A_an['z'].active = +(By_an(xg, yg, -dt/2)/(1j*kx)).real
A_an.copy_guards()

# Set initial magnetic field perturbation using the vector potential
B = Field(manifold, dtype=Float3)
manifold.curl(A_an, B, down=False)
# Add background magnetic field
B['x'].active += Bx
B['y'].active += By
B['z'].active += Bz
B.copy_guards()

div = Field(manifold, dtype=Float)
div.fill(0)
manifold.divergence(B, div)


# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge, eta=eta)

# Initialize state
state = State(ions, B)

# Initialize timestepper
e = TimeStepper(state, ohm, manifold)

# Deposit charges and calculate initial electric field
e.prepare(dt)

# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):  # noqa: E302
    return np.concatenate(comm.allgather(arr))


# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings

    global_B = concatenate(e.B.trim())
    global_B_an = concatenate((B_an(t=0)).trim())

    if comm.rank == 0:
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=2)
        im1 = axes[0].plot(xg[0, :], global_B['y'][0, :], 'b',
                           xg[0, :], global_B_an['y'][0, :], 'k--')
        im2 = axes[1].plot(xg[0, :], global_B['z'][0, :], 'b',
                           xg[0, :], global_B_an['z'][0, :], 'k--')
        axes[0].set_title(r'$B_y$')
        axes[1].set_title(r'$B_z$')
        axes[0].set_ylim(-A, A)
        axes[1].set_ylim(-A, A)
        for ax in axes:
            ax.set_xlim(0, Lx)

# Compute square of Fourier amplitude by projecting the local density
ampl2 = []
time = []
##########################################################################
# Main loop over time                                                    #
##########################################################################
for it in range(nt):

    # The update is handled by the experiment class
    e.iterate(dt)

    # if (it % 40): print(e.t)
    ampl2 += [(S*e.B['z'].trim()).sum()**2 + (C*e.B['z'].trim()).sum()**2]
    time += [e.t]

    # Make figures
    if plot:
        if (it % 200 == 0):
            global_B = concatenate(e.B.trim())
            global_B_an = concatenate((B_an(e.t)).trim())
            if comm.rank == 0:
                im1[0].set_ydata(global_B['y'][0, :])
                im1[1].set_ydata(global_B_an['y'][0, :])
                im2[0].set_ydata(global_B['z'][0, :])
                im2[1].set_ydata(global_B_an['z'][0, :])

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)
raise SystemExit
# Sum squared amplitude over processor, then take the square root
ampl = np.sqrt(comm.allreduce(ampl2, op=MPI.SUM))
# Convert list of times into NumPy array
time = np.array(time)

# Test if growth rate is correct
if comm.rank == 0:

    # Find first local maximum
    # index = argrelextrema(ampl, np.greater)
    tmax = time[0]
    ymax = ampl[0]

    # Theoretical gamma (TODO: Solve dispersion relation here)
    gamma_t = -0.48409456077745611*kx

    if plot or fitplot:
        import matplotlib.pyplot as plt
        # Create figure
        plt.figure(2)
        plt.clf()
        plt.semilogy(time, ampl, 'b')
        plt.semilogy(time, ymax*np.exp(gamma_t*(time - tmax)), 'k-')

        plt.title(r'$|\hat{\rho}(ikx=%d, iky=%d)|$' % (ikx, iky))
        plt.xlabel("time")
        # plt.savefig("landau-damping.pdf")
        plt.show()
