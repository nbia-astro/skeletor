from skeletor import Float, Float3, Field, Particles, Sources
from skeletor import Ohm
from skeletor.manifolds.second_order import Manifold
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

plot = False
# Interpolation order
order = 2
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
cs = np.sqrt(Te/mass)

vph = cs

# Total number of particles in simulation
N = npc*nx*ny


# Wave vector and its modulus
kx = 2*np.pi*ikx/Lx
ky = 2*np.pi*iky/Ly
k = np.sqrt(kx*kx + ky*ky)

# Frequency
omega = k*vph

# Time step
dt = 2**np.floor(np.log2(0.5*dx/vph))

# Simulation time
tend = 2*np.pi*nperiods/omega

# Number of time steps
nt = int(tend/dt)


def rho_an(x, y, t):
    """Analytic density as function of x, y and t"""
    return charge*(1 + A*np.cos(kx*x+ky*y)*np.sin(omega*t))


def ux_an(x, y, t):
    """Analytic x-velocity as function of x, y and t"""
    return -A*vph*np.sin(kx*x+ky*y)*np.cos(omega*t)*kx/k


def uy_an(x, y, t):
    """Analytic y-velocity as function of x, y and t"""
    return -A*vph*np.sin(kx*x+ky*y)*np.cos(omega*t)*ky/k


if quiet:
    # Uniform distribution of particle positions (quiet start)
    sqrt_npc = int(np.sqrt(npc))
    assert sqrt_npc**2 == npc
    npx = nx*sqrt_npc
    npy = ny*sqrt_npc
    x, y = np.meshgrid(
            Lx*(np.arange(npx) + 0.5)/npx,
            Ly*(np.arange(npy) + 0.5)/npy)
    x = x.flatten()
    y = y.flatten()
else:
    x = Lx*np.random.uniform(size=N).astype(Float)
    y = Ly*np.random.uniform(size=N).astype(Float)

# Perturbation to particle velocities
vx = ux_an(x, y, t=dt/2)
vy = uy_an(x, y, t=dt/2)
vz = np.zeros_like(vx)

# Add thermal velocity
vx += vtx*np.random.normal(size=N).astype(Float)
vy += vty*np.random.normal(size=N).astype(Float)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly, lbx=order, lby=order)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass, order=order)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, vz)

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.N, op=MPI.SUM) == N

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
assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
sources.add_guards()
assert np.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), N*charge/npc)
sources.copy_guards()

# Calculate electric field (Solve Ohm's law)
ohm(sources, B, E)
# Set boundary condition
E.copy_guards()


# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return np.concatenate(comm.allgather(arr))


# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings

    global_rho = concatenate(sources.rho.trim())
    global_E = concatenate(E.trim())
    global_rho_an = concatenate(rho_an(manifold.x, 0, 0))

    if comm.rank == 0:
        plt.rc('image', origin='lower', interpolation='nearest')
        plt.figure(1)
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
        vmin, vmax = charge*(1 - A), charge*(1 + A)
        im1, = ax1.plot(manifold.x, global_E['x'], 'b')
        im2 = ax2.plot(manifold.x, global_rho, 'b',
                       manifold.x, global_rho_an, 'k--')
        ax2.set_title(r'$\rho$')
        ax2.set_ylim((1-A), (1+A))
        ax1.set_ylim(-1.2e-4, 1.2e-4)

t = 0
diff2 = 0
# Big array for storing density at every time step
if comm.rank == 0:
    data = np.zeros((nx, nt))

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
    sources.add_guards()
    sources.copy_guards()

    # Calculate forces (Solve Ohm's law)
    ohm(sources, B, E)
    # Set boundary condition
    E.copy_guards()

    local_rho = sources.rho.trim()
    global_rho = concatenate(local_rho)
    global_E = concatenate(E.trim())
    if comm.rank == 0:
        data[:, it] = global_rho

    # Make figures
    if plot:
        if (it % 100 == 0):
            global_rho = concatenate(local_rho)
            global_rho_an = concatenate(rho_an(manifold.x, 0, t))
            if comm.rank == 0:
                im1.set_ydata(global_E['x'])
                im2[0].set_ydata(global_rho)
                im2[1].set_ydata(global_rho_an)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)

##########################################################################
# Analyze simulation                                                     #
##########################################################################
if comm.rank == 0:
    import matplotlib.pyplot as plt
    # Remove mean density
    data -= data.mean()

    # Wave numbers
    kx = 2*np.pi*np.fft.rfftfreq(nx)/manifold.dx

    # Frequency
    w = 2*np.pi*np.fft.rfftfreq(nt)/dt

    # Compute spacetime spectrum. Only show positive half of both frequency and
    # wavenumber spectra.
    spec = np.fft.rfft2(data.T)[:nt//2, :]

    dx = manifold.dx

    plt.rc('image', aspect='auto', interpolation='nearest', origin='lower')

    plt.figure(2)
    plt.clf()
    ikmin = 4
    plt.imshow((np.log(np.abs(spec[:, ikmin:])**2.)),
               extent=(kx[ikmin], kx[-1], w[0], w[-1]))

    from dispersion_solvers import IonacousticDispersion

    kxvec = np.linspace(1e-2, kx[-2], 100)

    solve_cic = IonacousticDispersion(Ti=1e-2, Te=Te, b=0, p=2, dx=dx)
    solve_tsc = IonacousticDispersion(Ti=1e-2, Te=Te, b=0, p=3, dx=dx)
    vph_cic = solve_cic.cold(kxvec)
    vph_tsc = solve_tsc.cold(kxvec)
    plt.plot(kxvec, kxvec*vph_cic, 'k:', label='CIC')
    plt.plot(kxvec, kxvec*vph_tsc, 'r:', label='TSC')
    plt.plot(kxvec, kxvec*cs, 'k--', label=r'$k c_s$')
    plt.ylim(0, 1.5*(kxvec*vph_cic.real).max())
    plt.xlabel(r'$k_x \Delta x$')
    plt.ylabel(r'$\omega$')
    plt.legend(frameon=False, loc='upper left')
    plt.savefig('test2.pdf')
    plt.show()
