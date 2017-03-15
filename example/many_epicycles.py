from skeletor import cppinit, Float, Float3, Grid, Field, Particles, Sources
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# Quiet start
quiet = True

# Plotting
plot = True

# Perturbation
perturb = True

# Time step
dt = 0.5e-3

# Simulation time
tend = 2*numpy.pi

# Number of time steps
nt = int(tend/dt)

t0 = -dt/2

# Particle charge and mass
charge = 1
mass = 1

# Keplerian frequency
Omega = 1

# Shear parameter
S = -3/2

# Magnetic field in z-direction
bz = 0.0

# Cycltron frequency in the z-direction
ocbz = charge/mass*bz

# Spin
Sz = ocbz + 2.0*Omega

# Gyration frequency
og = numpy.sqrt(Sz*(Sz + S))

# Phase
phi = numpy.pi/2

# Amplitude of perturbation
ampl = numpy.pi*5

# Number of grid points in x- and y-direction
nx, ny = 32, 64

# Average number of particles per cell
npc = 64

# Electron temperature
Te = 1.0

# Total number of particles in simulation
N = npc*nx*ny

if quiet:
    # Uniform distribution of particle positions (quiet start)
    sqrt_npc = int(numpy.sqrt(npc))
    assert sqrt_npc**2 == npc
    dx = dy = 1/sqrt_npc
    x, y = numpy.meshgrid(
            numpy.arange(dx/2, nx+dx/2, dx),
            numpy.arange(dy/2, ny+dy/2, dy))
    x0 = x.flatten()
    y0 = y.flatten()
else:
    x0 = nx*numpy.random.uniform(size=N).astype(Float)
    y0 = ny*numpy.random.uniform(size=N).astype(Float)


def y_an(t):
    y = ampl*numpy.cos(og*t + phi) + y0
    return y.astype(Float)


def x_an(t):
    x = +(Sz/og)*ampl*numpy.sin(og*t + phi) + x0 - S*t*(y0-ny/2)
    return x.astype(Float)


def vy_an(t):
    vy = -og*ampl*numpy.sin(og*t + phi)*numpy.ones(N)
    return vy.astype(Float)


def vx_an(t):
    vx = Sz*ampl*numpy.cos(og*t + phi) - S*(y0-ny/2)
    return vx.astype(Float)

# Particle position at t = -dt/2
x = x_an(-dt/2)
y = y_an(-dt/2)

# Particle velocity at t = 0
vx = vx_an(t=0)
vy = vy_an(t=0)

# Drift forward by dt/2
x += vx*dt/2
y += vy*dt/2


def shear_periodic(x, y, vx, vy, t, Lx, Ly):
    """Shearing periodic boundaries along y.

       This function modifies x and vx and subsequently applies periodic
       boundaries on x.
    """
    # Left to right
    x [y <  0.] -= S*Ly*t
    vx[y <  0.] -= S*Ly

    # Right to left
    x [y >= Ly] += S*Ly*t
    vx[y >= Ly] += S*Ly

    # Apply periodicity in y
    y[y <  0.] += Ly
    y[y >= Ly] -= Ly

    # Apply periodicity in x and y
    x[x  < 0.] += Lx
    x[x >= Lx] -= Lx

    return (x, y, vx, vy)

# Apply boundaries before intializing
(x, y, vx, vy) = shear_periodic(x, y, vx, vy, 0, nx, ny)

# Add perturbation
if perturb:
    kx = 2*numpy.pi/nx
    ky = 2*numpy.pi/ny
    vx += numpy.sin(kx*x + ky*y)

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
grid = Grid(nx, ny, comm)

# x- and y-grid
xg, yg = numpy.meshgrid(grid.x, grid.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
Nmax = int(1.25*N/nvp)

# Create particle array
ions = Particles(Nmax, charge, mass, Omega=Omega, S=S)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, grid)

# Make sure particles actually reside in the local subdomain
assert all(ions["y"][:ions.N] >= grid.edges[0])
assert all(ions["y"][:ions.N] < grid.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.N, op=MPI.SUM) == N

# Initialize sources
sources = Sources(grid, comm, dtype=Float)

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.N*charge)
sources.current.add_guards_ppic2()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), N*charge)

# Electric field in y-direction
E_star = Field(grid, comm, dtype=Float3)
E_star.fill((0.0, 0.0))

for i in range(nx+2):
    E_star['y'][:, i] = -2*S*(grid.yg-ny/2)*mass/charge*Omega


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))

# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings

    global_rho = concatenate(sources.rho.trim())

    if comm.rank == 0:
        plt.rc('image', origin='upper', interpolation='nearest')
        plt.figure(1)
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
        lines1 = ax1.plot(ions['y'][:N], ions['y'][:N], 'b.')
        lines2 = ax2.plot(ions['vx'][:N], ions['vy'][:N], 'b.')
        ax1.set_xlim(-1, ny+1)
        ax1.set_ylim(-1, nx+1)
        ax1.set_xlabel('y')
        ax1.set_ylabel('x')
        ax2.set_xlabel('vx')
        ax2.set_ylabel('vy')
        ax1.invert_yaxis()
        plt.figure(2)
        plt.clf()
        fig, ax1 = plt.subplots(num=2, ncols=1)
        im1 = ax1.imshow(global_rho)

t = 0
k = 0
##########################################################################
# Main loop over time                                                    #
##########################################################################
for it in range(nt):
    # Deposit sources
    sources.deposit(ions)
    assert numpy.isclose(sources.rho.sum(), ions.N*charge)
    sources.current.add_guards_ppic2()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), N*charge)

    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    ions.push(E_star, dt, t=t)

    assert comm.allreduce(ions.N, op=MPI.SUM) == N

    # Update time
    t += dt

    # Make figures
    if plot:
        if (it % 100 == 0):
            global_rho = concatenate(sources.rho.trim())
            if comm.rank == 0:
                lines1[0].set_data(ions['y'][:N], ions['x'][:N])
                # lines1[1].set_data(numpy.mod(y_an(t), ny), x_an(t))
                lines2[0].set_data(ions['vx'][:N], ions['vy'][:N])
                im1.set_data(global_rho)
                im1.autoscale()
                # lines2[1].set_data(vx_an(t), vy_an(t))
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                    plt.pause(1e-7)
