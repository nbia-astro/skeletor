from skeletor import Float, Float2, Field, Particles, Sources
from skeletor import Ohm, Faraday, InitialCondition
from skeletor.manifolds.second_order import Manifold
from skeletor.predictor_corrector import Experiment
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from dispersion_solvers import HallDispersion
from numpy import cos, sin, pi, arctan
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
npc = 2**16
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
cs = numpy.sqrt(Te/mass)

# Total number of particles in simulation
np = npc*nx*ny

# Wave vector and its modulus
kx = 2*numpy.pi*ikx/Lx
ky = 2*numpy.pi*iky/Ly
k = numpy.sqrt(kx*kx + ky*ky)

# Angle of k-vector with respect to x-axis
theta = arctan(iky/ikx) if ikx != 0 else pi/2

# Magnetic field strength
B0 = 1

(Bx, By, Bz) = (B0*cos(theta), B0*sin(theta), 0)
B2 = Bx**2 + By**2 + Bz**2

rho0 = 1.0

va = B0

# Ohmic resistivity
eta = 0

beta = 20

vt = numpy.sqrt(va**2*beta/2)
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

def frequency (kzva):
    hel = 1
    from numpy import sqrt
    return kzva*(sqrt (1.0 + (0.5*kzva/oc)**2) + 0.5*kzva/(hel*oc))

def get_dt(kzva):
    from numpy import pi, floor, log2
    dt = cfl/frequency (kzva)
    dt = 2.0**(floor (log2 (dt)))
    return dt

kmax = numpy.pi

vph = omega.real/kmax

# Simulation time
tend = 20.

# Phase factor
phase = lambda x, y, t: A*numpy.exp(1j*(omega*t - kx*x - ky*y))

# Linear solutions in real space
rho_an = lambda x, y, t: rho0 + rho0*(di.vec[m]['drho']*phase(x, y, t)).real
Bx_an = lambda x, y, t: Bx + B0*(di.vec[m]['bx']*phase(x, y, t)).real
By_an = lambda x, y, t: By + B0*(di.vec[m]['by']*phase(x, y, t)).real
Bz_an = lambda x, y, t: Bz + B0*(di.vec[m]['bz']*phase(x, y, t)).real
Ux_an = lambda x, y, t:         (di.vec[m]['vx']*phase(x, y, t)).real
Uy_an = lambda x, y, t:         (di.vec[m]['vy']*phase(x, y, t)).real
Uz_an = lambda x, y, t:         (di.vec[m]['vz']*phase(x, y, t)).real

# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(numpy.sqrt(npc))
assert sqrt_npc**2 == npc
a = (numpy.arange(sqrt_npc) + 0.5)/sqrt_npc
x_cell, y_cell = numpy.meshgrid(a, a)
x_cell = x_cell.flatten()
y_cell = y_cell.flatten()

R = (numpy.arange(npc) + 0.5)/npc
vx_cell = erfinv(2*R - 1)*numpy.sqrt(2)*vtx
vy_cell = erfinv(2*R - 1)*numpy.sqrt(2)*vty
vz_cell = erfinv(2*R - 1)*numpy.sqrt(2)*vtz
numpy.random.shuffle(vx_cell)
numpy.random.shuffle(vy_cell)
numpy.random.shuffle(vz_cell)
for i in range(nx):
    for j in range(ny):
        if i == 0 and j == 0:
            x = x_cell + i
            y = y_cell + j
            vx = vx_cell
            vy = vy_cell
            vz = vz_cell
        else:
            x = numpy.concatenate((x, x_cell + i))
            y = numpy.concatenate((y, y_cell + j))
            vx = numpy.concatenate((vx, vx_cell))
            vy = numpy.concatenate((vy, vy_cell))
            vz = numpy.concatenate((vz, vz_cell))

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, nlbx=1, nubx=1, nlby=1, nuby=1)

# Time step
# dt = cfl*manifold.dx/vph
dt = get_dt(kmax*va)

# Number of time steps
nt = int(tend/dt)

faraday = Faraday(manifold)

# x- and y-grid
xg, yg = numpy.meshgrid(manifold.x, manifold.y)

# Pair of Fourier basis functions with the specified wave numbers.
# The basis functions are normalized so that the Fourier amplitude can be
# computed by summing rather than averaging.
S = numpy.sin(kx*xg + ky*yg)/(nx*ny)
C = numpy.cos(kx*xg + ky*yg)/(nx*ny)

# Maximum number of electrons in each partition
npmax = int(1.5*np/comm.size)

# Create particle array
ions = Particles(manifold, npmax, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, vz)

# Create a uniform density field
# init = InitialCondition(npc, quiet=quiet, vt=vt)
# init(manifold, ions)

# Perturbation to particle velocities
ions['vx'] += Ux_an(ions['x'], ions['y'], t=-dt/2)
ions['vy'] += Uy_an(ions['x'], ions['y'], t=-dt/2)
ions['vz'] += Uz_an(ions['x'], ions['y'], t=-dt/2)

def B_an(t):
    B_an = Field(manifold, dtype=Float2)
    B_an['x'].active = Bx_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
    B_an['y'].active = By_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
    B_an['z'].active = Bz_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
    return B_an

# Create vector potential
A_an = Field(manifold, dtype=Float2)
A_an['x'].active = 0
A_an['y'].active = -((Bz + B0*(di.vec[m]['bz']*phase(xg, yg, -dt/2)))/
                    (1j*kx)).real
A_an['z'].active =  ((By + B0*(di.vec[m]['by']*phase(xg, yg, -dt/2)))/
                    (1j*kx)).real
A_an.copy_guards()

# Set initial magnetic field perturbation using the vector potential
B = Field(manifold, dtype=Float2)
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

# Initialize experiment
e = Experiment(manifold, ions, ohm, B, npc, io=None)

# Deposit charges and calculate initial electric field
e.prepare(dt)

# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return numpy.concatenate(comm.allgather(arr))

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
# Sum squared amplitude over processor, then take the square root
ampl = numpy.sqrt(comm.allreduce(ampl2, op=MPI.SUM))
# Convert list of times into NumPy array
time = numpy.array(time)

# Test if growth rate is correct
if comm.rank == 0:
    from scipy.signal import argrelextrema

    # Find first local maximum
    # index = argrelextrema(ampl, numpy.greater)
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
        plt.semilogy(time, ymax*numpy.exp(gamma_t*(time - tmax)), 'k-')

        plt.title(r'$|\hat{\rho}(ikx=%d, iky=%d)|$' % (ikx, iky))
        plt.xlabel("time")
        # plt.savefig("landau-damping.pdf")
        plt.show()

