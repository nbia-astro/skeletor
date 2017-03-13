from skeletor import Float, Float3, Field, Particles, Sources
from skeletor import Ohm
from skeletor.manifolds.second_order import Manifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

plot = True
# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 256, 1
# Grid size in x- and y-direction
Lx = 1
Ly = 2
dx = Lx/nx
dy = Ly/ny
# Average number of particles per cell
npc = 256
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 1.0
# Dimensionless amplitude of perturbation
A = 1.0e-5
# Wavenumbers
ikx = 1
iky = 0
# Thermal velocity of electrons in x- and y-direction
vtx, vty = 0.0, 0.0

# Number of periods to run for
nperiods = 4

# Sound speed
cs = numpy.sqrt(Te/mass)

vph = cs

# Total number of particles in simulation
np = npc*nx*ny


# Wave vector and its modulus
kx = 2*numpy.pi*ikx/Lx
ky = 2*numpy.pi*iky/Ly
k = numpy.sqrt(kx*kx + ky*ky)

# Frequency
omega = k*vph

# Time step
dt = 2**numpy.floor(numpy.log2(0.5*dx/vph))

# Simulation time
tend = 2*numpy.pi*nperiods/omega

# Number of time steps
nt = int(tend/dt)


def rho_an(x, y, t):
    """Analytic density as function of x, y and t"""
    return charge*(1 + A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t))


def ux_an(x, y, t):
    """Analytic x-velocity as function of x, y and t"""
    return -A*vph*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*kx/k


def uy_an(x, y, t):
    """Analytic y-velocity as function of x, y and t"""
    return -A*vph*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*ky/k

if quiet:
    # Uniform distribution of particle positions (quiet start)
    sqrt_npc = int(numpy.sqrt(npc))
    assert sqrt_npc**2 == npc
    npx = nx*sqrt_npc
    npy = ny*sqrt_npc
    x, y = numpy.meshgrid(
            Lx*(numpy.arange(npx) + 0.5)/npx,
            Ly*(numpy.arange(npy) + 0.5)/npy)
    x = x.flatten()
    y = y.flatten()
else:
    x = Lx*numpy.random.uniform(size=np).astype(Float)
    y = Ly*numpy.random.uniform(size=np).astype(Float)

# Perturbation to particle velocities
vx = ux_an(x, y, t=dt/2)
vy = uy_an(x, y, t=dt/2)
vz = numpy.zeros_like(vx)

# Add thermal velocity
vx += vtx*numpy.random.normal(size=np).astype(Float)
vy += vty*numpy.random.normal(size=np).astype(Float)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly)

# x- and y-grid
xg, yg = numpy.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
npmax = int(1.5*np/comm.size)

# Create particle array
ions = Particles(manifold, npmax, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, vz)

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.np, op=MPI.SUM) == np

# Set the electric field to zero
E = Field(manifold, dtype=Float3)
E.fill((0.0, 0.0, 0.0))
E.copy_guards()
B = Field(manifold, dtype=Float3)
B.fill((0.0, 0.0, 0.0))
B.copy_guards()


# Initialize sources
sources = Sources(manifold)

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge)

# Calculate initial density and force

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
sources.current.add_guards()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)
sources.current.copy_guards()

# Calculate electric field (Solve Ohm's law)
ohm(sources, B, E)
# Set boundary condition
E.copy_guards()


# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return numpy.concatenate(comm.allgather(arr))

# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings

    global_rho = concatenate(sources.rho.trim())
    global_E = concatenate(E.trim())
    global_rho_an = concatenate(rho_an(xg, yg, 0))

    if comm.rank == 0:
        plt.rc('image', origin='lower', interpolation='nearest')
        plt.figure(1)
        plt.clf()
        fig, (ax1, ax2, ax3) = plt.subplots(num=1, ncols=3)
        vmin, vmax = charge*(1 - A), charge*(1 + A)
        im1 = ax1.imshow(global_rho, vmin=vmin, vmax=vmax)
        im2, = ax2.plot(xg[0, :], global_E['x'][0, :], 'b')
        im3 = ax3.plot(xg[0, :], global_rho[0, :], 'b',
                       xg[0, :], global_rho_an[0, :], 'k--')
        ax1.set_title(r'$\rho$')
        # ax3.set_ylim(vmin, vmax)
        ax3.set_ylim((1-A), (1+A))
        ax2.set_ylim(-1.2e-4, 1.2e-4)

t = 0
diff2 = 0
# Big array for storing density at every time step
if comm.rank == 0:
    data = numpy.zeros((ny, nx, nt))

##########################################################################
# Main loop over time                                                    #
##########################################################################
for it in range(nt):
    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    ions.push(E, B, dt)

    # Update time
    t += dt

    # Deposit sources
    sources.deposit(ions)

    # Boundary calls
    sources.current.add_guards()
    sources.current.copy_guards()

    # Calculate forces (Solve Ohm's law)
    ohm(sources, B, E)
    # Set boundary condition
    E.copy_guards()

    local_rho = sources.rho.trim()
    global_rho = concatenate(local_rho)
    global_E = concatenate(E.trim())
    if comm.rank == 0:
        data[:, :, it] = global_rho

    # Make figures
    if plot:
        if (it % 100 == 0):
            global_rho = concatenate(local_rho)
            global_rho_an = concatenate(rho_an(xg, yg, t))
            if comm.rank == 0:
                im1.set_data(global_rho)
                im2.set_ydata(global_E['x'][0, :])
                im3[0].set_ydata(global_rho[0, :])
                im3[1].set_ydata(global_rho_an[0, :])
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)

##########################################################################
# Analyze simulation                                                     #
##########################################################################
if comm.rank == 0:
    from numpy.fft import rfftfreq, rfft2
    import matplotlib.pyplot as plt
    # Remove mean density
    data -= data.mean()

    # Wave numbers
    kx = 2*numpy.pi*rfftfreq(nx)/manifold.dx

    # Frequency
    w = 2*numpy.pi*rfftfreq(nt)/dt

    # Average out y dimension (for now)
    sli = numpy.mean(data, axis=0).T

    ikx_min = 32
    # Compute spacetime spectrum. Only show positive half of both frequency and
    # wavenumber spectra.
    spec = rfft2(sli)[:nt//2, :]

    dx = manifold.dx

    plt.rc('image', aspect='auto', interpolation='nearest')

    plt.figure(2)
    plt.clf()
    plt.imshow((numpy.log(numpy.abs(spec)**2.)),
               extent=(kx[0], kx[-1], w[0], w[-1]))

    from dispersion_solvers import IonacousticDispersion

    solve = IonacousticDispersion(Ti=1e-2, Te=Te, b=0, p=2, dx=dx)
    kxvec = numpy.linspace(1e-2, kx[-2], 100)
    vph = solve.cold(kxvec)
    plt.plot(kxvec, kxvec*vph, 'k')
    plt.plot(kxvec, kxvec*cs, 'k--')
    plt.ylim(0, 1.5*(kxvec*vph.real).max())
    plt.xlabel(r'$k_x \Delta x$')
    plt.ylabel(r'$\omega$')
    # plt.savefig('test.pdf')
    plt.show()
