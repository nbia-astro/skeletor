from skeletor import Float, Particles, Sources
from skeletor.manifolds.second_order import ShearingManifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
import matplotlib.pyplot as plt

# Initial time of particle positions
t = -numpy.pi/3

# Particle charge and mass
charge = 1
mass = 1

# Keplerian frequency
Omega = 0

# Shear parameter
S = -3/2

# Amplitude of perturbation
ampl = 2.

# Number of grid points in x- and y-direction
nx, ny = 32, 32

# Average number of particles per cell
npc = 16

# Total number of particles in simulation
np = npc*nx*ny

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = ShearingManifold(nx, ny, comm, Lx=nx, Ly=ny, S=S, Omega=Omega)

# x- and y-grid
xx, yy = numpy.meshgrid(manifold.x, manifold.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
npmax = int(5*np/comm.size)

# Create particle array
ions = Particles(manifold, npmax, time=t, charge=charge, mass=mass)

# Lagrangian/labeling coordinates
# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(numpy.sqrt(npc))
assert sqrt_npc**2 == npc, "'npc' must be a square of an integer."
ax, ay = [ab.flatten().astype(Float) for ab in numpy.meshgrid(
    manifold.dx*(numpy.arange(nx*sqrt_npc) + 0.5)/sqrt_npc,
    manifold.dy*(numpy.arange(ny*sqrt_npc) + 0.5)/sqrt_npc
    )]

# x-component of wave vector
kx = 2*numpy.pi/nx


def velocity(a):
    """Particle velocity in Lagrangian coordinates."""
    return ampl*numpy.sin(kx*a)


def vx_an(a, b, t):
    """Particle velocity along x is perturbation plus shear"""
    vx = velocity(a) - b*S
    return vx


def x_an(a, b, t):
    """Particle x-position as a function of time"""
    return a + vx_an(a, b, t)*t


# Particle position (i.e. Eulerian coordinate) and velocity
x = x_an(ax, ay, t)
y = ay
vx = vx_an(ax, ay, t)
vy = numpy.zeros_like(vx)
vz = numpy.zeros_like(vx)

# Assign particles to subdomains (zero velocity and uniform distribution)
ions.initialize(x, y, vx, vy, vz)

# Set boundary condition on particles
ions.shear_periodic_y()
ions.periodic_x()

# Make sure particles actually reside in the local subdomain
assert all(ions["y"][:ions.np] >= manifold.edges[0])
assert all(ions["y"][:ions.np] < manifold.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.np, op=MPI.SUM) == np

# Initialize sources
sources = Sources(manifold, npc)

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
sources.rho.add_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)
sources.rho.copy_guards()


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))


global_rho = concatenate(sources.rho.trim())
global_J = concatenate(sources.J.trim())

plt.rc('image', origin='lower', interpolation='nearest', cmap='coolwarm')
plt.figure(1)
plt.clf()
plt.plot(manifold.x, (global_J['x']/global_rho).mean(axis=0))
plt.figure(2)
plt.clf()
plt.imshow(global_J['x']/global_rho)
plt.show()
