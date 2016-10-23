from skeletor import cppinit, Float, Float2, Grid, Particles, Sources
from skeletor import ShearField
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# Quiet start
quiet = True

# Plotting
plot = True

# Time step
dt = 0.5e-3

# Simulation time
tend = 2*numpy.pi

# Number of time steps
nt = int(tend/dt)

# Particle charge and mass
charge = 1
mass = 1

# Keplerian frequency
Omega = 1

# Shear parameter
S = -3/2

# Epicyclic frequency
kappa = numpy.sqrt(2*Omega*(2*Omega+S))

# Amplitude of perturbation
ampl = 0.1

# Number of grid points in x- and y-direction
nx, ny = 32, 64

# Average number of particles per cell
npc = 16

# Wave numbers
kx = 2*numpy.pi/nx
ky0 = 0

# Total number of particles in simulation
np = npc*nx*ny

if quiet:
    # Uniform distribution of particle positions (quiet start)
    sqrt_npc = int(numpy.sqrt(npc))
    assert sqrt_npc**2 == npc
    dx = dy = 1/sqrt_npc
    x, y = numpy.meshgrid(
            numpy.arange(dx/2, nx+dx/2, dx),
            numpy.arange(dy/2, ny+dy/2, dy))
    x = x.flatten()
    y = y.flatten()
else:
    x = nx*numpy.random.uniform(size=np).astype(Float)
    y = ny*numpy.random.uniform(size=np).astype(Float)

# Particle velocity at t = 0
vx = -S*y
vy = numpy.zeros_like(x)

# Add perturbation
vx += ampl*numpy.cos(kx*x + ky0*y)
vy += 2*Omega/kappa*ampl*numpy.sin(kx*x + ky0*y)

# Drift forward by dt/2
x += vx*dt/2
y += vy*dt/2

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
grid = Grid(nx, ny, comm)

# x- and y-grid
xx, yy = numpy.meshgrid(grid.x, grid.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
npmax = int(1.25*np/nvp)

# Create particle array
ions = Particles(npmax, charge, mass, Omega=Omega, S=S)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, grid)

# Make sure particles actually reside in the local subdomain
assert all(ions["y"][:ions.np] >= grid.edges[0])
assert all(ions["y"][:ions.np] < grid.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.np, op=MPI.SUM) == np

# Initialize sources
sources = Sources(grid, comm, dtype=Float)
sources.rho = ShearField(grid, comm, dtype=Float)
rho_periodic = ShearField(grid, comm, dtype=Float)

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.np*charge)
sources.rho.add_guards(St=0)
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), np*charge)


def theta(a, t, phi=0):
    return kx*a - kappa*t + phi


def ux(a, t):
    return ampl*numpy.exp(1j*theta(a, t))


def uy(a, t):
    return 2*Omega/(1j*kappa)*ux(a, t)


def uxp(a, t):
    return ux(a, t) + S*t*uy(a, t)


def xp(a, t):
    A = a - 1/(1j*kappa)*(uxp(a, t) - uxp(a, 0))
    B = S/kappa**2*(uy(a, t) - uy(a, 0))
    return A + B


def alpha(a, t):
    y = 1 + 1j*kx*(xp(a, t) - a)
    return y


def rho(a, t):
    return numpy.real(1/alpha(a, t))

a = grid.x

# Electric field in y-direction
E_star = ShearField(grid, comm, dtype=Float2)
E_star.fill((0.0, 0.0))

for i in range(nx+2):
    E_star['y'][:, i] = -2*S*grid.yg*mass/charge*Omega


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))


# Make initial figure
if plot:
    import matplotlib.pyplot as plt

    global_rho = concatenate(sources.rho[:grid.nyp+1, :nx+1])

    if comm.rank == 0:
        plt.rc('image', origin='upper', interpolation='nearest',
               cmap='coolwarm')
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=2)
        im1 = axes[0].imshow(global_rho)
        im2 = axes[1].imshow(global_rho)
        plt.figure(2)
        plt.clf()
        fig2, ax1 = plt.subplots(num=2, ncols=1)
        im4 = ax1.plot(grid.x, (rho_periodic[:grid.nyp, :nx].
                                mean(axis=0)-npc)/npc, 'b',
                       xp(a, 0), rho(a, 0) - 1, 'r')
        ax1.set_ylim(-4*ampl, 4*ampl)

t = dt/2

##########################################################################
# Main loop over time                                                    #
##########################################################################

for it in range(nt):
    # Deposit sources
    sources.deposit(ions)
    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards(S*t)
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)

    sources.rho.copy_guards(S*t)
    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    ions.push(E_star, dt, t=t)

    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Update time
    t += dt

    # Copy density into a shear field
    rho_periodic[:grid.nyp, :nx] = sources.rho.trim()

    # Translate the density to be periodic in y
    rho_periodic.translate(-S*t)
    rho_periodic.copy_guards(0)

    # Make figures
    if plot:
        if (it % 60 == 0):
            if comm.rank == 0:
                im1.set_data(sources.rho[:grid.nyp+1, :nx+1])
                im2.set_data(rho_periodic[:grid.nyp+1, :nx+1])
                im1.autoscale()
                im2.autoscale()
                im4[0].set_ydata((rho_periodic[:grid.nyp, :nx].
                                 mean(axis=0)-npc)/npc)
                im4[1].set_data(xp(a, t), rho(a, t) - 1)
                plt.pause(1e-7)
