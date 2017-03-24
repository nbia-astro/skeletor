from skeletor import Float3, Field, Particles
from skeletor import Ohm, Faraday, State
from skeletor.manifolds.second_order import Manifold
from skeletor.time_steppers.horowitz import TimeStepper
import numpy as np
from numpy.random import normal
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from scipy.special import erfinv

fitplot = True
plot = False
# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 32, 1
# Grid size in x- and y-direction (square cells!)
Lx, Ly = 16.758881558993764, 16.758881558993764

# Grid distances
dx, dy = Lx/nx, Ly/ny

# Average number of particles per cell
npc = 2**10
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 1.0
# Dimensionless amplitude of perturbation
A = 0.005
# Wavenumbers
ikx = 1
iky = 0
ampl = 1e-6
# Thermal velocity of electrons in x- and y-direction

# Magnetic field strength
B0 = 1
va = B0
beta_para = 4
beta_perp = 1
vtx = np.sqrt(beta_para/2)
vty = np.sqrt(beta_perp/2)
vtz = vty

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

(Bx, By, Bz) = (B0*np.cos(theta), B0*np.sin(theta), 0)

# Simulation time
tend = 40

# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(np.sqrt(npc))
assert sqrt_npc**2 == npc
a = (np.arange(sqrt_npc) + 0.5)/sqrt_npc
x_cell, y_cell = np.meshgrid(a, a)
x_cell = x_cell.flatten()
y_cell = y_cell.flatten()

if comm.rank == 0:
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
else:
    x, y, vx, vy, vz = None, None, None, None, None

x = comm.bcast(x, root=0)
y = comm.bcast(y, root=0)
vx = comm.bcast(vx, root=0)
vy = comm.bcast(vy, root=0)
vz = comm.bcast(vz, root=0)

x *= dx
y *= dy

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly, lbx=2, lby=2)

# Time step
dt = 1e-2

# Number of time steps
nt = int(tend/dt)

faraday = Faraday(manifold)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, vz)

# Perturbation to magnetic field
# Add background magnetic field
B = Field(manifold, dtype=Float3)
B.fill((Bx, By, Bz))
B['x'] += ampl*normal(size=B['x'].shape)
B['y'] += ampl*normal(size=B['y'].shape)
B['z'] += ampl*normal(size=B['z'].shape)
B.copy_guards()


# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge)

# Initialize state
state = State(ions, B)

# Initialize timestepper
e = TimeStepper(state, ohm, manifold)

# Deposit charges and calculate initial electric field
e.prepare(dt)


# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return np.concatenate(comm.allgather(arr))

if plot or fitplot:
    import matplotlib.pyplot as plt

# Make initial figure
if plot:
    from matplotlib.cbook import mplDeprecation
    import warnings

    global_B = concatenate(e.B.trim())

    if comm.rank == 0:
        plt.rc('image', origin='lower', cmap='RdYlBu')
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=3)
        im1 = axes[0].imshow(global_B['x'])
        im2 = axes[1].imshow(global_B['y'])
        im3 = axes[2].imshow(global_B['z'])
        axes[0].set_title(r'$B_x$')
        axes[1].set_title(r'$B_y$')
        axes[2].set_title(r'$B_z$')

##########################################################################
# Main loop over time                                                    #
##########################################################################
Bx_mag = []
By_mag = []
Bz_mag = []
time = []
for it in range(nt):

    # The update is handled by the experiment class
    e.iterate(dt)

    # Make figures
    if (it % 20 == 0):
        print(e.t)
        global_B = concatenate(e.B.trim())
        if comm.rank == 0:
            Bx_mag.append(((global_B['x']-Bx)**2).mean())
            By_mag.append(((global_B['y']-By)**2).mean())
            Bz_mag.append(((global_B['z']-Bz)**2).mean())
            time.append(e.t)
            if plot:
                im1.set_data(global_B['x'])
                im2.set_data(global_B['y'])
                im3.set_data(global_B['z'])
                for im in (im1, im2, im3):
                    im.autoscale()

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)
if comm.rank == 0:
    if fitplot:

        Bx_mag = np.sqrt(np.array(Bx_mag))
        By_mag = np.sqrt(np.array(By_mag))
        Bz_mag = np.sqrt(np.array(Bz_mag))
        B_mag = np.sqrt(Bx_mag**2 + By_mag**2 + Bz_mag**2)
        time = np.array(time)

        # TODO: Solve this here!
        gamma_t = 0.18894406899999999

        from scipy.optimize import curve_fit

        # Exponential growth function
        def func(x, a, b):
            return a*np.exp(b*x)

        def lin_func(x, a, b):
            return a + b*x

        # Fit exponential to the evolution of sqrt(mean(B_x**2))
        # Disregard first half of data
        first = int(0.5*nt/20)
        last = int(0.8*nt/20)
        popt, pcov = curve_fit(lin_func, time[first:last],
                               np.log(B_mag[first:last]))
        plt.figure(2)
        plt.semilogy(time, Bx_mag, label=r"$|\delta B_x|$")
        plt.semilogy(time, By_mag, label=r"$|\delta B_y|$")
        plt.semilogy(time, Bz_mag, label=r"$|\delta B_z|$")
        plt.semilogy(time, B_mag, label=r"$|\delta B|$")
        gamma_f = popt[1]
        plt.semilogy(time, func(time-3, 1e-6, popt[1]), '--',
                     label=r"Fit: $\gamma = %.5f$" % gamma_f)
        plt.semilogy(time, func(time-3, 1e-6, gamma_t), 'k-',
                     label=r"Theory: $\gamma = %.5f$" % gamma_t)
        plt.legend(frameon=False, loc=2)
        # plt.savefig("parallel-firehose.pdf")
        err = (gamma_t - gamma_f)/gamma_t
        print("Relative error: {}".format(err))
        plt.xlabel(r"$t$")
        plt.show()
