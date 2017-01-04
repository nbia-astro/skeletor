from skeletor import cppinit, Float, Particles, Sources
from skeletor import ShearField
from skeletor.manifolds.second_order import ShearingManifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
import matplotlib.pyplot as plt
import matplotlib.widgets as mw

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
npc = 64

# Wave numbers
kx = 2*numpy.pi/nx

# Total number of particles in simulation
np = npc*nx*ny

# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(numpy.sqrt(npc))
assert sqrt_npc**2 == npc
dx = dy = 1/sqrt_npc
a, b = numpy.meshgrid(
        numpy.arange(dx/2, nx+dx/2, dx),
        numpy.arange(dy/2, ny+dy/2, dy))
a = a.flatten()
b = b.flatten()


def x_an(ap, bp, t):
    phi = kx*ap
    x = 2*Omega/kappa*ampl*(numpy.sin(kappa*t + phi) - numpy.sin(phi)) + ap \
        - S*t*(bp - ampl*numpy.cos(phi))
    return x


def y_an(ap, bp, t):
    phi = kx*ap
    y = ampl*(numpy.cos(kappa*t + phi) - numpy.cos(phi)) + bp
    return y


def alpha_particle(ap, t):
    phi = kx*ap
    dxda = 2*Omega/kappa*ampl*kx*(numpy.cos(kappa*t + phi) - numpy.cos(phi)) \
        + 1 - S*t*ampl*kx*numpy.sin(phi)
    dyda = -ampl*kx*(numpy.sin(kappa*t + phi) - numpy.sin(phi))

    return dxda + S*t*dyda


def rho_an_particle(a, t):
    return 1/alpha_particle(a, t)


# Phase
phi = kx*a

# Particle positions at time= 0
x = a
y = b

vx = numpy.zeros_like(x)
vy = numpy.zeros_like(y)

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = ShearingManifold(nx, ny, comm, S=S, Omega=Omega)

# x- and y-grid
xx, yy = numpy.meshgrid(manifold.x, manifold.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
npmax = int(5*np/nvp)

# Create particle array
ions = Particles(manifold, npmax, time=0, charge=charge, mass=mass)

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
sources.rho = ShearField(manifold, time=0, dtype=Float)
rho_periodic = ShearField(manifold, time=0, dtype=Float)

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.np*charge)
sources.rho.add_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), np*charge)
sources.rho.copy_guards()


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))


def update(t):

    ions['x'][:np] = x_an(a, b, t)
    ions['y'][:np] = y_an(a, b, t)
    ions.time = t
    ions.shear_periodic_y()
    ions.periodic_x()

    # Deposit sources
    sources.deposit(ions)
    sources.rho.time = t

    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards()

    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)

    sources.rho.copy_guards()

    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Copy density into a shear field
    rho_periodic.active = sources.rho.trim()

    # Translate the density to be periodic in y
    rho_periodic.translate(-t)
    rho_periodic.copy_guards()

    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(rho_periodic.trim())
    if comm.rank == 0:
        im1a.set_data(global_rho)
        im2a.set_data(global_rho_periodic)
        im1a.autoscale()
        im2a.autoscale()
        im4[1].set_ydata(global_rho_periodic.mean(axis=0)/npc)
        xp_par = x_an(a, b, t) + S*y_an(a, b, t)*t
        xp_par %= nx
        im4[0].set_data(xp_par, rho_an_particle(a, t))


if comm.rank == 0:
    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(rho_periodic.trim())

    plt.rc('image', origin='upper', interpolation='nearest',
           cmap='coolwarm')
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, nrows=2)
    im1a = axes[0].imshow(global_rho)
    im2a = axes[1].imshow(global_rho_periodic)
    axtime1 = plt.axes([0.125, 0.1, 0.775, 0.03])
    stime1 = mw.Slider(axtime1, 'Time', -numpy.pi, numpy.pi/2, 0)
    stime1.on_changed(update)

    plt.figure(2)
    plt.clf()
    fig2, ax1 = plt.subplots(num=2)
    plt.subplots_adjust(bottom=0.25)
    ax1.set_ylim(0, 2)
    ax1.set_xlabel(r"$x'$")
    ax1.set_title(r'$\rho/\rho_0$')
    # Create slider widget for changing time
    axtime2 = plt.axes([0.125, 0.1, 0.775, 0.03])
    stime2 = mw.Slider(axtime2, 'Time', -numpy.pi, numpy.pi/2, 0)
    stime2.on_changed(update)
    xp_par = x_an(a, b, 0) + S*y_an(a, b, 0)*0
    xp_par %= nx
    xp_par = numpy.sort(xp_par)
    im4 = ax1.plot(xp_par, rho_an_particle(xp_par, 0), 'k-',
                   manifold.x, (global_rho_periodic.mean(axis=0))/npc, 'r-')
    update(0)
    plt.show()
