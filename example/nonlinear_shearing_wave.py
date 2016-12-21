from skeletor import cppinit, Float, Float2, Particles, Sources
from skeletor import ShearField
from skeletor.manifolds.second_order import ShearingManifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# Quiet start
quiet = True

# Plotting
plot = True

# Time step
dt = 0.5e-3

# Initial time of particle positions
t = dt/2

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
nx, ny = 32, 16

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
manifold = ShearingManifold(nx, ny, comm, S=S, Omega=Omega)

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
sources.rho = ShearField(manifold, time=t, dtype=Float)
rho_periodic = ShearField(manifold, time=0, dtype=Float)
J_periodic = ShearField(manifold, time=0, dtype=Float2)

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.np*charge)
sources.rho.add_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), np*charge)
sources.rho.copy_guards()

sources.J.add_guards_vector()
sources.J.copy_guards()


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


a = manifold.x

# Electric field
E = ShearField(manifold, dtype=Float2)
E.fill((0.0, 0.0))


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))


# Make initial figure
if plot:
    import matplotlib.pyplot as plt

    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(rho_periodic.trim())
    global_J = concatenate(sources.J.trim())
    global_J_periodic = concatenate(J_periodic.trim())

    if comm.rank == 0:
        plt.rc('image', origin='upper', interpolation='nearest',
               cmap='coolwarm')
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=2, nrows=3)
        im1a = axes[0, 0].imshow(global_rho)
        im2a = axes[0, 1].imshow(global_rho_periodic)
        im1b = axes[1, 0].imshow(global_J['x']/global_rho)
        im2b = axes[1, 1].imshow(global_J_periodic['x']/global_rho_periodic)
        im1c = axes[2, 0].imshow(global_J['y']/global_rho)
        im2c = axes[2, 1].imshow(global_J_periodic['y']/global_rho_periodic)
        plt.figure(2)
        plt.clf()
        fig2, (ax1, ax2, ax3) = plt.subplots(num=2, nrows=3)
        im4 = ax1.plot(manifold.x, (global_rho_periodic.mean(axis=0))/npc,
                       'b', xp(a, 0), rho(a, 0), 'r--')
        im5 = ax2.plot(manifold.x, (global_J_periodic['x']/global_rho_periodic)
                       .mean(axis=0), 'b', xp(a, 0), ux(a, 0), 'r--')
        im6 = ax3.plot(manifold.x, (global_J_periodic['y']/global_rho_periodic)
                       .mean(axis=0), 'b', xp(a, 0), uy(a, 0), 'r--')
        ax1.set_ylim(1 - 4*ampl, 1 + 4*ampl)
        ax2.set_ylim(-4*ampl, 4*ampl)
        ax3.set_ylim(-4*ampl, 4*ampl)

##########################################################################
# Main loop over time                                                    #
##########################################################################

for it in range(nt):
    # Deposit sources
    sources.deposit(ions)
    sources.rho.time = t
    sources.J.time = t
    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards()
    sources.J.add_guards_vector()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)

    sources.rho.copy_guards()
    sources.J.copy_guards()

    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    ions.push_modified(E, dt)

    # Update time
    t += dt

    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Copy density into a shear field
    rho_periodic.active = sources.rho.trim()
    J_periodic.active = sources.J.trim()

    # Translate the density to be periodic in y
    rho_periodic.translate(-t)
    rho_periodic.copy_guards()

    J_periodic.translate_vector(-t)
    J_periodic.copy_guards()

    # Make figures
    if plot:
        if (it % 60 == 0):
            global_rho = concatenate(sources.rho.trim())
            global_rho_periodic = concatenate(rho_periodic.trim())
            global_J = concatenate(sources.J.trim())
            global_J_periodic = concatenate(J_periodic.trim())
            if comm.rank == 0:
                im1a.set_data(global_rho)
                im2a.set_data(global_rho_periodic)
                im1b.set_data(global_J['x']/global_rho)
                im2b.set_data(global_J_periodic['x']/global_rho_periodic)
                im1c.set_data(global_J['y']/global_rho)
                im2c.set_data(global_J_periodic['y']/global_rho_periodic)
                im1a.autoscale()
                im2a.autoscale()
                im1b.autoscale()
                im2b.autoscale()
                im1c.autoscale()
                im2c.autoscale()
                im4[0].set_ydata(global_rho_periodic.mean(axis=0)/npc)
                im5[0].set_ydata((global_J_periodic['x']/global_rho_periodic).
                                 mean(axis=0))
                im6[0].set_ydata((global_J_periodic['y']/global_rho_periodic).
                                 mean(axis=0))
                im4[1].set_data(xp(a, t), rho(a, t))
                im5[1].set_data(xp(a, t), ux(a, t))
                im6[1].set_data(xp(a, t), uy(a, t))
                plt.pause(1e-7)
