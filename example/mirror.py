from skeletor import Float2, Field, Particles
from skeletor import Ohm, Faraday, InitialCondition
from skeletor.manifolds.second_order import Manifold
from skeletor.predictor_corrector import Experiment
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from numpy import cos, sin, pi, arctan

plot = True
# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 32, 32
# Grid size in x- and y-direction (square cells!)
Lx = nx
Ly = Lx*ny/nx
# Average number of particles per cell
npc = 1024
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 0.0
# Dimensionless amplitude of perturbation
A = 0.005
# Wavenumbers
ikx = 1
iky = 0
# Thermal velocity of electrons in x- and y-direction
vtx, vty, vtz = 4., 12., 12.

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

# Simulation time
tend = 10

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, nlbx=1, nubx=1, nlby=1, nuby=1)

# Time step
dt = 5e-3

# Number of time steps
nt = int(tend/dt)

faraday = Faraday(manifold)

# x- and y-grid
xg, yg = numpy.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
npmax = int(1.5*np/comm.size)

# Create particle array
ions = Particles(manifold, npmax, charge=charge, mass=mass)

# Create a uniform density field
init = InitialCondition(npc, quiet=quiet)
init(manifold, ions)

# Perturbation to particle velocities
from numpy.random import normal
ions['vx'][:ions.np] = vtx*normal(size=ions.np)
ions['vy'][:ions.np] = vty*normal(size=ions.np)
ions['vz'][:ions.np] = vtz*normal(size=ions.np)

# Add background magnetic field
B = Field(manifold, dtype=Float2)
B.fill((Bx, By, Bz))
B.copy_guards()


# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge)

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
    if plot:
        Bx_mag = numpy.array(Bx_mag)
        By_mag = numpy.array(By_mag)
        By_mag = numpy.array(By_mag)
        time = numpy.array(time)
        plt.figure(2)
        plt.semilogy(time, Bx_mag, label=r"$(\delta B_x)^2$")
        plt.semilogy(time, By_mag, label=r"$(\delta B_y)^2$")
        plt.semilogy(time, Bz_mag, label=r"$(\delta B_z)^2$")
        plt.legend(frameon=False, loc=2)
        plt.xlabel(r"$t$")
        plt.show()