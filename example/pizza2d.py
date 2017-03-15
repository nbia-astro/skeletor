from skeletor import cppinit, Float, Float3, Particles, Sources
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
N = npc*nx*ny

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


def vx_an(ap, bp, t):
    phi = kx*ap
    vx = 2*Omega*ampl*numpy.cos(kappa*t + phi) - S*(bp - ampl*numpy.cos(phi))
    return vx


def vy_an(ap, bp, t):
    phi = kx*ap
    vy = -ampl*kappa*numpy.sin(kappa*t + phi)
    return vy


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
Nmax = int(5*N/nvp)

# Create particle array
ions = Particles(manifold, Nmax, time=0, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy)

# Make sure particles actually reside in the local subdomain
assert all(ions["y"][:ions.N] >= manifold.edges[0])
assert all(ions["y"][:ions.N] < manifold.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.N, op=MPI.SUM) == N

# Initialize sources
sources = Sources(manifold)
sources.rho = ShearField(manifold, time=0, dtype=Float)
rho_periodic = ShearField(manifold, time=0, dtype=Float)
Jx_periodic = ShearField(manifold, time=0, dtype=Float3)
Jy_periodic = ShearField(manifold, time=0, dtype=Float3)

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.N*charge)
sources.current.add_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), N*charge)
sources.current.copy_guards()


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))


def update(t):

    ions['x'][:N] = x_an(a, b, t)
    ions['y'][:N] = y_an(a, b, t)
    ions['vx'][:N] = vx_an(a, b, t)
    ions['vy'][:N] = vy_an(a, b, t)
    ions.time = t
    ions.shear_periodic_y()
    ions.periodic_x()

    # Deposit sources
    sources.deposit(ions)
    sources.current.time = t

    assert numpy.isclose(sources.rho.sum(), ions.N*charge)
    sources.current.add_guards()

    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), N*charge)

    sources.current.copy_guards()

    assert comm.allreduce(ions.N, op=MPI.SUM) == N

    # Copy density into a shear field
    rho_periodic.active = sources.rho.trim()
    Jx_periodic.active = sources.Jx.trim()
    Jy_periodic.active = sources.Jy.trim()

    # Translate the density to be periodic in y
    rho_periodic.translate(-t)
    rho_periodic.copy_guards()

    Jx_periodic.translate(-t)
    Jx_periodic.copy_guards()

    Jy_periodic.translate(-t)
    Jy_periodic.copy_guards()

    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(rho_periodic.trim())
    global_Jx = concatenate(sources.Jx.trim())
    global_Jx_periodic = concatenate(Jx_periodic.trim())
    global_Jy = concatenate(sources.Jy.trim())
    global_Jy_periodic = concatenate(Jy_periodic.trim())
    if comm.rank == 0:
        im1a.set_data(global_rho)
        im2a.set_data(global_rho_periodic)
        im1b.set_data(global_Jx/global_rho)
        im2b.set_data(global_Jx_periodic/global_rho_periodic)
        im1c.set_data(global_Jy/global_rho)
        im2c.set_data(global_Jy_periodic/global_rho_periodic)
        im1a.autoscale()
        im2a.autoscale()
        im1b.autoscale()
        im2b.autoscale()
        im1c.autoscale()
        im2c.autoscale()
        im4[0].set_ydata(global_rho_periodic.mean(axis=0)/npc)
        im5[0].set_ydata((global_Jx_periodic/global_rho_periodic).
                         mean(axis=0))
        im6[0].set_ydata((global_Jy_periodic/global_rho_periodic).
                         mean(axis=0))
        xp_par = x_an(manifold.x, 0, t) + S*y_an(manifold.x, 0, t)*t
        xp_par %= nx
        ind = numpy.argsort(xp_par)
        im4[1].set_data(xp_par[ind], rho_an_particle(manifold.x, t)[ind])
        im6[1].set_data(xp_par[ind], vy_an(manifold.x, 0, t)[ind])
        im5[1].set_data(xp_par[ind], vx_an(manifold.x, 0, t)[ind]+S*y_an
                        (manifold.x, 0, t)[ind])


if comm.rank == 0:
    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(rho_periodic.trim())
    global_Jx = concatenate(sources.Jx.trim())
    global_Jx_periodic = concatenate(Jx_periodic.trim())
    global_Jy = concatenate(sources.Jy.trim())
    global_Jy_periodic = concatenate(Jy_periodic.trim())

    plt.rc('image', origin='upper', interpolation='nearest',
           cmap='coolwarm')
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=2, nrows=3)
    im1a = axes[0, 0].imshow(global_rho)
    im2a = axes[0, 1].imshow(global_rho_periodic)
    im1b = axes[1, 0].imshow(global_Jx/global_rho)
    im2b = axes[1, 1].imshow(global_Jx_periodic/global_rho_periodic)
    im1c = axes[2, 0].imshow(global_Jy/global_rho)
    im2c = axes[2, 1].imshow(global_Jy_periodic/global_rho_periodic)
    axtime1 = plt.axes([0.125, 0.1, 0.775, 0.03])
    stime1 = mw.Slider(axtime1, 'Time', -numpy.pi, numpy.pi/2, 0)
    stime1.on_changed(update)

    plt.figure(2)
    plt.clf()
    fig2, (ax1, ax2, ax3) = plt.subplots(num=2, nrows=3)
    plt.subplots_adjust(bottom=0.25)
    ax1.set_ylim(0, 2)
    ax2.set_ylim(-1*ampl, 1*ampl)
    ax3.set_ylim(-2*ampl, 2*ampl)
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, nx)
    ax1.set_xlabel(r"$x'$")
    ax1.set_title(r'$\rho/\rho_0$')
    # Create slider widget for changing time
    axtime2 = plt.axes([0.125, 0.1, 0.775, 0.03])
    stime2 = mw.Slider(axtime2, 'Time', -numpy.pi, numpy.pi/2, 0)
    stime2.on_changed(update)
    xp_par = x_an(a, b, 0) + S*y_an(a, b, 0)*0
    xp_par %= nx
    xp_par = numpy.sort(xp_par)
    im4 = ax1.plot(manifold.x, (global_rho_periodic.mean(axis=0))/npc,
                   'b',
                   manifold.x, (global_rho_periodic.mean(axis=0))/npc,
                   'r--')
    im5 = ax2.plot(manifold.x, (global_Jx_periodic/global_rho_periodic)
                   .mean(axis=0), 'b',
                   manifold.x, (global_Jx_periodic/global_rho_periodic)
                   .mean(axis=0), 'r--')
    im6 = ax3.plot(manifold.x, (global_Jy_periodic/global_rho_periodic)
                   .mean(axis=0), 'b',
                   manifold.x, (global_Jy_periodic/global_rho_periodic)
                   .mean(axis=0), 'r--')
    update(0)
    plt.show()
