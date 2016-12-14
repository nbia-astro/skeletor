from skeletor import Float, Particles, Sources, cppinit
from skeletor.manifolds.ppic2 import Manifold
from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy

# Number of grid points in x- and y-direction
nx, ny = 16, 16

# Average number of particles
npc = 32

# Particle charge and mass
charge = 1.0
mass = 1.0

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid with weird setup of ghost layers
manifold = Manifold(nx, ny, comm, nlbx=1, nubx=2, nlby=3, nuby=4)

# # Initialize sources
sources = Sources(manifold, dtype=Float)

# Total number of particles
np = npc*nx*ny
# Maximum number of particles in each partition
npmax = int(1.5*np/nvp)

# Create particle array
particles = Particles(manifold, npmax, charge=charge, mass=mass)

# Synchronize random number generator across ALL processes
numpy.random.set_state(comm.bcast(numpy.random.get_state()))

# Uniform distribution of particle positions
x = nx*numpy.random.uniform(size=np).astype(Float)
y = ny*numpy.random.uniform(size=np).astype(Float)
# Normal distribution of particle velocities
vx = numpy.empty(np, Float)
vy = numpy.empty(np, Float)

# Assign particles to subdomains
particles.initialize(x, y, vx, vy)

# deposit particles
sources.deposit(particles)
assert(numpy.isclose(sources.rho.sum(), charge*particles.np))
sources.rho.add_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.sum(), op=SUM), np*charge)
sources.rho.copy_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=SUM), np*charge)

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.imshow(sources.rho, interpolation='nearest', origin='lower')
# sources.rho.add_guards()
# plt.figure(2)
# plt.imshow(sources.rho, interpolation='nearest', origin='lower')
# sources.rho.copy_guards()
# plt.figure(3)
# plt.imshow(sources.rho, interpolation='nearest', origin='lower')
# plt.show()
