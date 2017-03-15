from skeletor import Float, Particles, Sources
from skeletor.manifolds.ppic2 import Manifold
from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy as np

# Number of grid points in x- and y-direction
nx, ny = 16, 16

# Average number of particles
npc = 32

# Particle charge and mass
charge = 1.0
mass = 1.0

# Create numerical grid with weird setup of ghost layers
manifold = Manifold(nx, ny, comm, lbx=1, lby=2)

# # Initialize sources
sources = Sources(manifold)

# Total number of particles
N = npc*nx*ny
# Maximum number of particles in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
particles = Particles(manifold, Nmax, charge=charge, mass=mass)

# Synchronize random number generator across ALL processes
np.random.set_state(comm.bcast(np.random.get_state()))

# Uniform distribution of particle positions
x = manifold.Lx*np.random.uniform(size=N).astype(Float)
y = manifold.Ly*np.random.uniform(size=N).astype(Float)
# Normal distribution of particle velocities
vx = np.empty(N, Float)
vy = np.empty(N, Float)
vz = np.empty(N, Float)

# Assign particles to subdomains
particles.initialize(x, y, vx, vy, vz)


def test_extended_grid():
    # deposit particles
    sources.deposit(particles)
    assert(np.isclose(sources.rho.sum(), charge*particles.N/npc))
    sources.current.add_guards()
    assert np.isclose(comm.allreduce(
        sources.rho.sum(), op=SUM), N*charge/npc)
    sources.current.copy_guards()
    assert np.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=SUM), N*charge/npc)

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.imshow(sources.rho, interpolation='nearest', origin='lower')
# sources.current.add_guards()
# plt.figure(2)
# plt.imshow(sources.rho, interpolation='nearest', origin='lower')
# sources.current.copy_guards()
# plt.figure(3)
# plt.imshow(sources.rho, interpolation='nearest', origin='lower')
# plt.show()
