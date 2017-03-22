"""
This script creates a widget comparing the density field for a sheared
disturbance with the analytical solution. This disturbance correponds to
the standard 1D Burgers' equation in the primed coordinate, x' = x + Sty. See
also test_burgers.py.
"""

from skeletor import cppinit, Float, Float3, Particles, Sources, ShearField
from skeletor.manifolds.second_order import ShearingManifold
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
import matplotlib.pyplot as plt
import matplotlib.widgets as mw

# Particle charge and mass
charge = 1
mass = 1

# Keplerian frequency
Omega = 0

# Shear parameter
S = -3/2

# Amplitude of perturbation
ampl = 0.2

# Number of grid points in x- and y-direction
nx, ny = 64, 128

Lx, Ly = 1, 2

dx, dy = Lx/nx, Ly/ny

# Average number of particles per cell
npc = 64

# Wave numbers
kx = 2*np.pi/Lx

# Total number of particles in simulation
N = npc*nx*ny

# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(np.sqrt(npc))
assert sqrt_npc**2 == npc
dx1 = dy1 = 1/sqrt_npc
a, b = np.meshgrid(
        np.arange(dx1/2, nx+dx1/2, dx1),
        np.arange(dy1/2, ny+dy1/2, dy1))
a = a.flatten()
b = b.flatten()

a *= dx
b *= dy

def mean(f, axis=None):
    """Compute mean of an array across processors."""
    result = np.mean(f, axis=axis)
    if axis is None or axis == 0:
        # If the mean is to be taken over *all* axes or just the y-axis,
        # then we need to communicate
        result = comm.allreduce(result, op=MPI.SUM)/comm.size
    return result


def rms(f):
    """Compute root-mean-square of an array across processors."""
    return np.sqrt(mean(f**2))


def velocity(a):
    """Particle velocity in Lagrangian coordinates."""
    return ampl*np.sin(kx*a)


def velocity_prime(a):
    """Derivative of particle velocity in Lagrangian coordinates:
    ∂v(a,τ)/∂a"""
    return ampl*kx*np.cos(kx*a)


def euler(a, τ):
    """
    This function converts from Lagrangian to Eulerian sheared coordinate x'
    by solving ∂x'(a, τ)/∂τ = U(a) for x'(a, τ) subject to the initial
    condition x'(a, 0) = a.
    """
    return (a + velocity(a)*τ) % Lx


def euler_prime(a, τ):
    """
    The derivative ∂x'/∂a of the conversion function defined above, which is
    related to the mass density in Lagrangian coordinates through
    rho(a, τ)/rho_0(a) = (∂x'/∂a)⁻¹, where rho_0(a) = rho(a, 0) is the initial
    mass density.
    """
    return 1 + velocity_prime(a)*τ


def lagrange(xp, t, tol=1.48e-8, maxiter=50):
    """
    Given the Eulerian coordinate x' = x + Sty and time t, this function
    solves the definition x' = euler(a, t) for the Lagrangian coordinate a via
    the Newton-Raphson method.
    """
    # Use Eulerian coordinate as initial guess
    a = xp.copy()
    for it in range(maxiter):
        f = euler(a, t) - xp
        df = euler_prime(a, t)
        b = a - f/df
        # This is not the safest criterion, but seems good enough
        if rms(a - b) < tol:
            return b
        a = b.copy()


def rho_an(a, t):
    """
    Calculates the analytical density as a function of the lagragian
    coordiate, a.
    """
    return 1/euler_prime(a, t)


def rho2d_an(x, y, t):
    """
    Calculate the analytical density as a function of Eulerian grid position,
    (x, y), and time, t. Accepts 2D arrays for x and y.
    """
    xp = x + S*y*t
    xp %= Lx
    a = lagrange(xp, t)
    return rho_an(a, t)


def vx_an(a, b, t):
    """Particle velocity along x is perturbation plus shear"""
    vx = velocity(a) - b*S
    return vx


def x_an(a, b, t):
    """Particle x-position as a function of time"""
    return a + vx_an(a, b, t)*t


# Particle positions at time= 0
x = a
y = b

vx = np.zeros_like(x)
vy = np.zeros_like(y)
vz = np.zeros_like(y)

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = ShearingManifold(nx, ny, comm, S=S, Omega=Omega, Lx=Lx, Ly=Ly)

# x- and y-grid
xx, yy = np.meshgrid(manifold.x, manifold.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
Nmax = int(5*N/nvp)

# Create particle array
ions = Particles(manifold, Nmax, time=0, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, vz)

# Make sure particles actually reside in the local subdomain
assert all(ions["y"][:ions.N] >= manifold.edges[0])
assert all(ions["y"][:ions.N] < manifold.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.N, op=MPI.SUM) == N

# Initialize sources
sources = Sources(manifold)
sources_periodic = Sources(manifold, time=0)

# Deposit sources
sources.deposit(ions)
assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
sources.current.add_guards()
assert np.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), N*charge/npc)
sources.current.copy_guards()

# Copy density into a shear field
sources_periodic.rho.active = sources.rho.trim()


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return np.concatenate(comm.allgather(arr))


def update(t):
    """
    This function updates the particle position and velocity according to the
    analytical solution, deposits the charge and current, and produces two
    figures. Figure 1 shows rho and Jx, which are shearing periodic, in
    the first column. In the second column it shows the same fields but
    translated by a distance -S*t*y along x. This makes the fields periodic.

    The second figures then shows the analytical solutions for rho and Jx
    as a function of the primed coordinate, x', which are compared with the
    numerical solution found by averaging the periodic fields shown in Figure
    1 along y.
    """

    ions['x'][:N] = x_an(a, b, t)/manifold.dx
    ions['vx'][:N] = vx_an(a, b, t)
    ions.time = t
    ions.shear_periodic_y()
    ions.periodic_x()

    # Deposit sources
    sources.deposit(ions)
    sources.current.time = t

    assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
    sources.current.add_guards()

    assert np.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), N*charge/npc)

    sources.current.copy_guards()

    assert comm.allreduce(ions.N, op=MPI.SUM) == N

    # Copy density into a shear field
    sources_periodic.rho.active = sources.rho.trim()
    sources_periodic.Jx.active = sources.Jx.trim()

    # Translate the density to be periodic in y
    sources_periodic.rho.translate(-t)
    sources_periodic.rho.copy_guards()

    sources_periodic.Jx.translate(-t)
    sources_periodic.Jx.copy_guards()

    err = rms(sources.rho.trim() - rho2d_an(xx, yy, t))

    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(sources_periodic.rho.trim())
    global_Jx = concatenate(sources.Jx.trim())
    global_Jx_periodic = concatenate(sources_periodic.Jx.trim())

    if comm.rank == 0:
        # Update 2D images
        im1a.set_data(global_rho)
        im2a.set_data(global_rho_periodic)
        im1b.set_data(global_Jx/global_rho)
        im2b.set_data(global_Jx_periodic/global_rho_periodic)
        for im in (im1a, im2a, im1b, im2b):
            im.autoscale()
        # Update 1D solutions (numerical and analytical)
        im4[0].set_ydata(global_rho_periodic.mean(axis=0))
        im5[0].set_ydata((global_Jx_periodic/global_rho_periodic).
                         mean(axis=0))
        xp = euler(manifold.x, t)
        im4[1].set_data(xp, rho_an(manifold.x, t))
        im5[1].set_data(xp, velocity(manifold.x))

        print(err)


global_rho = concatenate(sources.rho.trim())
global_rho_periodic = concatenate(sources_periodic.rho.trim())
global_Jx = concatenate(sources.Jx.trim())
global_Jx_periodic = concatenate(sources.Jx.trim())

if comm.rank == 0:

    plt.rc('image', origin='upper', interpolation='nearest',
           cmap='coolwarm')
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=2, nrows=2, sharex=True, sharey=True)
    for axis in axes.flatten():
        # This is necessary to remove unwanted white space when sharing axes
        axis.set_adjustable('box-forced')
    fig.subplots_adjust(bottom=0.15, top=0.95)
    im1a = axes[0, 0].imshow(global_rho)
    im2a = axes[0, 1].imshow(global_rho_periodic)
    im1b = axes[1, 0].imshow(global_Jx/global_rho)
    im2b = axes[1, 1].imshow(global_Jx_periodic/global_rho_periodic)
    axtime1 = plt.axes([0.125, 0.05, 0.775, 0.03])
    stime1 = mw.Slider(axtime1, 'Time', -np.pi/4, np.pi/8, 0)
    stime1.on_changed(update)

    plt.figure(2)
    plt.clf()
    fig2, (ax1, ax2) = plt.subplots(num=2, nrows=2, sharex=True)
    plt.subplots_adjust(bottom=0.2, top=0.9)
    ax1.set_ylim(0, 2)
    ax2.set_ylim(-1*ampl, 1*ampl)
    for ax in (ax1, ax2):
        ax.set_xlim(0, Lx)
    ax2.set_xlabel(r"$x'$")
    ax1.set_title(r'$\rho/\rho_0$')
    # Create slider widget for changing time
    axtime2 = plt.axes([0.125, 0.05, 0.775, 0.03])
    stime2 = mw.Slider(axtime2, 'Time', -np.pi/4, np.pi/8, 0)
    stime2.on_changed(update)
    xp = euler(manifold.x, 0)
    xp = np.sort(xp)
    im4 = ax1.plot(manifold.x, (global_rho_periodic.mean(axis=0)),
                   'b',
                   manifold.x, (global_rho_periodic.mean(axis=0)),
                   'r--')
    im5 = ax2.plot(manifold.x, (global_Jx_periodic/global_rho_periodic)
                   .mean(axis=0), 'b',
                   manifold.x, (global_Jx_periodic/global_rho_periodic)
                   .mean(axis=0), 'r--')
    update(0)
    plt.show()
