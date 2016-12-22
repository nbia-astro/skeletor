from skeletor import cppinit, Float, Float2, Particles, Sources
from skeletor import Field
from skeletor.manifolds.second_order import Manifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# Quiet start
quiet = True

# Plotting
visualization = True

# Time step
dt = 1e-2

# Initial time of particle positions
t = - 100

# Simulation time
tend = 120

# Number of time steps
nt = int((tend-t)/dt)

# Particle charge and mass
charge = 1
mass = 1

# Amplitude of perturbation
ampl = 0.1

# Number of grid points in x- and y-direction
nx, ny = 64, 1

# Average number of particles per cell
npc = 16

# Wave numbers
kx = 2*numpy.pi/nx
ky = 0

# Total number of particles in simulation
np = npc*nx*ny

if quiet:
    # Uniform distribution of particle positions (quiet start)
    sqrt_npc = int(numpy.sqrt(npc))
    assert sqrt_npc**2 == npc
    dx = dy = 1/sqrt_npc
    a, b = numpy.meshgrid(
            numpy.arange(dx/2, nx+dx/2, dx),
            numpy.arange(dy/2, ny+dy/2, dy))
    a = a.flatten()
    b = b.flatten()
else:
    a = nx*numpy.random.uniform(size=np).astype(Float)
    b = ny*numpy.random.uniform(size=np).astype(Float)

# Particle velocity at t = 0
vx = ampl*numpy.sin(kx*a)
vy = numpy.zeros_like(a)

# Start the positions at time = t
x = numpy.mod(a + vx*t, nx)
y = b


# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm)

# x- and y-grid
xx, yy = numpy.meshgrid(manifold.x, manifold.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
npmax = int(1.25*np/nvp)

# Create particle array
ions = Particles(manifold, npmax, time=t, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy)

# Make sure particles actually reside in the local subdomain
assert all(ions["y"][:ions.np] >= manifold.edges[0])
assert all(ions["y"][:ions.np] < manifold.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.np, op=MPI.SUM) == np

# Initialize sources
sources = Sources(manifold)

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.np*charge)
sources.rho.add_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), np*charge)


def ux(a):
    return ampl*numpy.sin(kx*a)


def xp(a, t):
    A = a + ux(a)*t
    return A


def rho(a, t):
    return 1/(1 + ampl*kx*numpy.cos(kx*a)*t)

a = manifold.x

# Electric field
E = Field(manifold, dtype=Float2)
E.fill((0.0, 0.0))


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))


# Make initial figure
if visualization:
    import matplotlib.pyplot as plt

    global_rho = concatenate(sources.rho.trim())
    global_J = concatenate(sources.J.trim())

    if comm.rank == 0:
        plt.rc('image', origin='upper', interpolation='nearest',
               cmap='coolwarm')
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=2)
        im1 = axes[0].imshow(global_rho)
        im2 = axes[1].imshow(global_J['x'])
        plt.figure(2)
        plt.clf()
        fig2, (ax1, ax2) = plt.subplots(num=2, nrows=2)
        im3 = ax1.plot(manifold.x, global_rho.mean(axis=0)/npc, 'b-',
                       xp(a, t), rho(a, t), 'r--')
        im4 = ax2.plot(manifold.x, ((global_J['x']/global_rho).mean(axis=0)),
                       'b', xp(a, 0), ux(a), 'r--')
        ax1.set_ylim(0, 3)

##########################################################################
# Main loop over time                                                    #
##########################################################################

for it in range(nt):
    # Deposit sources
    sources.deposit(ions)

    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards()
    sources.J.add_guards_vector()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)

    sources.rho.copy_guards()
    sources.J.copy_guards()

    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    ions.push(E, dt)

    # Update time
    t += dt

    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Make figures
    if visualization:
        if (it % 60 == 0):
            global_rho = concatenate(sources.rho.trim())
            global_J = concatenate(sources.J.trim())
            if comm.rank == 0:
                im1.set_data(global_rho)
                im2.set_data(global_J['x'])
                im1.autoscale()
                im2.autoscale()
                im3[0].set_ydata(global_rho.mean(axis=0)/npc)
                im3[1].set_data(xp(a, t), rho(a, t))
                im4[0].set_ydata(((global_J['x']/global_rho)).mean(axis=0))
                im4[1].set_data(xp(a, t), ux(a))
                plt.pause(1e-7)
