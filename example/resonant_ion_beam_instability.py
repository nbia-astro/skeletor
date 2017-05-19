from skeletor import Float3, Field, Particles
from skeletor import Ohm, InitialCondition, State
from skeletor.manifolds.second_order import Manifold
from skeletor.time_steppers.horowitz import TimeStepper
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
plot = False
fitplot = True
# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 64, 1
# Grid size in x- and y-direction (square cells!)
Lx = 49.3244179861
Ly = 1
# Average number of particles per cell
npc = 64
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 0.0
# Wavenumbers
ikx = 1
iky = 1

# CFL number
cfl = 0.1
# Number of periods to run for
nperiods = 1.0

# Sound speed
cs = np.sqrt(Te/mass)

# Total number of particles in simulation
N = npc*nx*ny

# Wave vector and its modulus
kx = 2*np.pi*ikx/Lx
ky = 2*np.pi*iky/Ly
k = np.sqrt(kx*kx + ky*ky)

(Bx, By, Bz) = (1, 0, 0)
B2 = Bx**2 + By**2 + Bz**2
va2 = B2

v0 = -0.5263157894736842
vb = 10
n0 = 0.95
nb = 0.05

# Simulation time
tend = 80.

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly)

# Time step
dt = cfl*manifold.dx/vb

# Number of time steps
nt = int(tend/dt)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions1 = Particles(manifold, Nmax, charge=charge, mass=mass, n0=n0)

# Create a uniform density field
init = InitialCondition(npc, quiet=quiet)
init(manifold, ions1)

# Perturbation to particle velocities
ions1['vx'] = v0

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions1.N, op=MPI.SUM) == N

##########################################################################
# Create a second, identical, ion array
# Create particle array
ions2 = Particles(manifold, Nmax, charge=charge, mass=mass, n0=nb)

# Create a uniform density field
init = InitialCondition(npc, quiet=quiet, vt=1e-4)
init(manifold, ions2)

# Perturbation to particle velocities
ions2['vx'] = vb

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions2.N, op=MPI.SUM) == N

##########################################################################
# Set the magnetic field to zero
B = Field(manifold, dtype=Float3)
B.fill((Bx, By, Bz))
B.copy_guards()

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge)

# Initialize state
state = State([ions1, ions2], B)

# Initialize experiment
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
    # e.iterate(dt)
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
        gamma_t = 0.298685440518

        from scipy.optimize import curve_fit

        # Exponential growth function
        def func(x, a, b):
            return a*np.exp(b*x)

        def lin_func(x, a, b):
            return a + b*x

        # Fit exponential to the evolution of sqrt(mean(B_x**2))
        # Disregard some of the data
        first = int(0.25*nt/20)
        last = int(0.6*nt/20)
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
        # plt.savefig("resonant_ion_beam_instability.pdf")
        err = (gamma_t - gamma_f)/gamma_t
        print("Relative error: {}".format(err))
        plt.xlabel(r"$t$")
        plt.show()
