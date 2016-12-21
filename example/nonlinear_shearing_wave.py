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
t = 0

# Simulation time
tend = numpy.pi/2

# Number of time steps
nt = int((tend-t)/dt)

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
ampl = 2.

# Number of grid points in x- and y-direction
nx, ny = 64, 64

# Average number of particles per cell
npc = 16

# Wave numbers
kx = 2*numpy.pi/nx

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


def x_an(ap, bp, t):
    phi = kx*ap
    x = 2*Omega/kappa*ampl*(numpy.sin(kappa*t + phi) - numpy.sin(phi)) + ap \
        - S*t*(bp - ampl*numpy.cos(phi))
    return x


def y_an(ap, bp, t):
    phi = kx*ap
    y = ampl*(numpy.cos(kappa*t + phi) - numpy.cos(phi)) + bp
    return y


def vx_an(ap, bp, t):
    phi = kx*ap
    vx = 2*Omega*ampl*numpy.cos(kappa*t + phi) - S*(bp - ampl*numpy.cos(phi))
    return vx


def vy_an(ap, bp, t):
    phi = kx*ap
    vy = -ampl*kappa*numpy.sin(kappa*t + phi)
    return vy


def alpha_particle(a, t):
    phi = kx*a
    dxda = 2*Omega/kappa*ampl*kx*(numpy.cos(kappa*t + phi) - numpy.cos(phi)) \
        + 1 - S*t*ampl*kx*numpy.sin(phi)
    dyda = -ampl*kx*(numpy.sin(kappa*t + phi) - numpy.sin(phi))

    return dxda + S*t*dyda


def rho_an_particle(a, t):
    return 1/alpha_particle(a, t)


# Phase
phi = kx*a

# Particle velocities at time = t-dt/2
vx = vx_an(a, b, t-dt/2)
vy = vy_an(a, b, t-dt/2)

# Particle positions at time=t
x = x_an(a, b, t)
y = y_an(a, b, t)

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
ions = Particles(manifold, npmax, time=dt/2, charge=charge, mass=mass)

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

ag = manifold.x

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
                       'b',
                       manifold.x, (global_rho_periodic.mean(axis=0))/npc,
                       'r--')
        im5 = ax2.plot(manifold.x, (global_J_periodic['x']/global_rho_periodic)
                       .mean(axis=0), 'b',
                       manifold.x, (global_J_periodic['x']/global_rho_periodic)
                       .mean(axis=0), 'r--')
        im6 = ax3.plot(manifold.x, (global_J_periodic['y']/global_rho_periodic)
                       .mean(axis=0), 'b',
                       manifold.x, (global_J_periodic['y']/global_rho_periodic)
                       .mean(axis=0), 'r--')
        ax1.set_ylim(0.5, 1.8)
        ax2.set_ylim(-1*ampl, 1*ampl)
        ax3.set_ylim(-2*ampl, 2*ampl)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(0, nx)

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
                xp_par = x_an(ag, 0, t) + S*y_an(ag, 0, t)*t
                xp_par %= nx
                ind = numpy.argsort(xp_par)
                im4[1].set_data(xp_par[ind], rho_an_particle(ag, t)[ind])
                im5[1].set_data(xp_par[ind], vx_an(ag, 0, t)[ind]
                                + S*y_an(ag, 0, t)[ind])
                im6[1].set_data(xp_par[ind], vy_an(ag, 0, t)[ind])

                plt.pause(1e-7)
