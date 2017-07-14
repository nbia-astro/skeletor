from skeletor import Float, Particles, Sources
from skeletor.manifolds.second_order import ShearingManifold
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
import matplotlib.pyplot as plt

# Initial time of particle positions
t = -np.pi/3

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
N = npc*nx*ny

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = ShearingManifold(nx, ny, comm,
                            Lx=nx, Ly=ny, x0=-nx/2, y0=-ny/2,
                            S=S, Omega=Omega, lbx=2, lby=2)

# x- and y-grid
xx, yy = np.meshgrid(manifold.x, manifold.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
Nmax = int(5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, time=t, charge=charge, mass=mass, order=2)

# Lagrangian/labeling coordinates
# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(np.sqrt(npc))
assert sqrt_npc**2 == npc, "'npc' must be a square of an integer."
ax, ay = [ab.flatten().astype(Float) for ab in np.meshgrid(
    manifold.dx*(np.arange(nx*sqrt_npc) + 0.5)/sqrt_npc,
    manifold.dy*(np.arange(ny*sqrt_npc) + 0.5)/sqrt_npc
    )]

# x-component of wave vector
kx = 2*np.pi/nx


def velocity(a):
    """Particle velocity in Lagrangian coordinates."""
    return ampl*np.sin(kx*a)


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
vy = np.zeros_like(vx)
vz = np.zeros_like(vx)

# Assign particles to subdomains (zero velocity and uniform distribution)
N = 2
x = np.array([-1, -1+nx-1e-8])
y = np.array([-1, -1+ny-1e-8])
vx = np.ones_like(x)
vy = np.ones_like(x)
vz = np.zeros_like(x)
ions.initialize(x, y, vx, vy, vz)
ions.N = 2
ions["x"][:ions.N] = x
ions["y"][:ions.N] = y
ions["vx"][:ions.N] = vx
ions["vy"][:ions.N] = vy
ions["vz"][:ions.N] = vz

# Set boundary condition on particles
# ions.shear_periodic_y()
# ions.periodic_x()

# Make sure particles actually reside in the local subdomain
# assert all(ions["y"][:ions.N] >= manifold.edges[0])
# assert all(ions["y"][:ions.N] < manifold.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
# assert comm.allreduce(ions.N, op=MPI.SUM) == N

# Initialize sources
sources = Sources(manifold)
sources.time = t

# Deposit sources
sources.deposit(ions)
# assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
# sources.add_guards()
# assert np.isclose(comm.allreduce(
#     sources.rho.trim().sum(), op=MPI.SUM), N*charge/npc)
# sources.copy_guards()


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return np.concatenate(comm.allgather(arr))


# global_rho = concatenate(sources.rho.trim())
# global_Jx = concatenate(sources.Jx.trim())

plt.rc('image', origin='lower', interpolation='nearest', cmap='coolwarm')
plt.figure(1)
plt.clf()
plt.imshow(sources.rho)
plt.figure(2)
plt.clf()
plt.imshow(sources.Jx)
plt.figure(3)
plt.clf()
plt.imshow(sources.Jy)
plt.show()
