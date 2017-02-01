from skeletor import Float, Float2, Field, Particles, Sources
from skeletor import Ohm, Faraday
from skeletor.manifolds.second_order import Manifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

plot = True

# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 64, 64
# Grid size in x- and y-direction
Lx = nx
Ly = Lx*ny/nx
dx = Lx/nx
dy = Ly/ny
# Average number of particles per cell
npc = 64
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 0.0
# Dimensionless amplitude of perturbation
A = 1e-4
# Wavenumbers
ikx = 1
iky = 1
# Thermal velocity of electrons in x- and y-direction
vtx, vty = 0.0, 0.0
# CFL number
cfl = 0.1
# Number of periods to run for
nperiods = 1.0

# Sound speed
cs = numpy.sqrt(Te/mass)

# Total number of particles in simulation
np = npc*nx*ny

# Wave vector and its modulus
kx = 2*numpy.pi*ikx/Lx
ky = 2*numpy.pi*iky/Ly
k = numpy.sqrt(kx*kx + ky*ky)

(Bx, By, Bz) = (0, 0, 1)
B2 = Bx**2 + By**2 + Bz**2
va2 = B2

vph = numpy.sqrt(cs*2 + va2)

# Frequency
omega = k*vph

# Time step
dt = cfl*dx/vph

# Simulation time
tend = 2*numpy.pi*nperiods/omega

# Number of time steps
nt = int(tend/dt)

def rho_an(x, y, t):
    """Analytic density as function of x, y and t"""
    return npc*charge*(1 + A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t))

def Bz_an(x, y, t):
    """Analytic density as function of x, y and t"""
    return Bz*(1 + A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t))

def ux_an(x, y, t):
    """Analytic x-velocity as function of x, y and t"""
    return -omega/k*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*kx/k

def uy_an(x, y, t):
    """Analytic y-velocity as function of x, y and t"""
    return -omega/k*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*ky/k

def uz_an(x, y, t):
    """Analytic z-velocity as function of x, y and t"""
    return -omega/k*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*0

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
vx = ux_an(x, y, t=0)
vy = uy_an(x, y, t=0)
vz = uz_an(x, y, t=0)

# Add thermal velocity
vx += vtx*numpy.random.normal(size=np).astype(Float)
vy += vty*numpy.random.normal(size=np).astype(Float)

x += vx*dt/2
y += vy*dt/2

x = numpy.mod(x, Lx)
y = numpy.mod(y, Ly)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, nlbx=1, nubx=1, nlby=1, nuby=1,
                    Lx=Lx, Ly=Ly)

faraday = Faraday(manifold)

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
E = Field(manifold, dtype=Float2)
E.fill((0.0, 0.0, 0.0))

# Set the magnetic field to zero
B = Field(manifold, dtype=Float2)
B.fill((Bx, By, Bz))
# B.copy_guards()

# Initialize sources
sources = Sources(manifold)

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge, npc=npc)

# Calculate initial density and force

# Deposit sources
sources.deposit(ions)
assert numpy.isclose(sources.rho.sum(), ions.np*charge)
sources.rho.add_guards()
sources.J.add_guards_vector()
assert numpy.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), np*charge)
sources.rho.copy_guards()
sources.J.copy_guards()

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
    global_rho_an = concatenate(rho_an(xg, yg, 0))
    global_B = concatenate(B.trim())
    global_Bz_an = concatenate(Bz_an(xg, yg, 0))

    if comm.rank == 0:
        plt.rc('image', origin='lower', interpolation='nearest')
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols=3, nrows=3)
        vmin, vmax = charge*(1 - A), charge*(1 + A)
        im1 = axes[0,0].imshow(global_rho, vmin=vmin, vmax=vmax)
        im2 = axes[0,1].plot(xg[0, :], global_B['z'][0, :], 'b',
                             xg[0, :], global_Bz_an[0, :], 'k--')
        im3 = axes[0,2].plot(xg[0, :], global_rho[0, :]/npc, 'b',
                       xg[0, :], global_rho_an[0, :]/npc, 'k--')
        im4 = axes[1,0].imshow(B['x'])
        im5 = axes[1,1].imshow(B['y'])
        im6 = axes[1,2].imshow(B['z'])
        im7 = axes[2,0].imshow(E['x'])
        im8 = axes[2,1].imshow(E['y'])
        im9 = axes[2,2].imshow(E['z'])
        # ax1.set_title(r'$\rho$')
        axes[0,2].set_ylim(vmin, vmax)
        axes[0,1].set_ylim(vmin, vmax)
        axes[0,2].set_xlim(0, x[-1])

t = 0
diff2 = 0
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
    sources.rho.add_guards()
    sources.J.add_guards_vector()
    sources.rho.copy_guards()
    sources.J.copy_guards()

    # Calculate forces (Solve Ohm's law)
    ohm(sources, B, E)
    E.copy_guards()

    faraday(E, B, dt)
    B.copy_guards()

    # Difference between numerical and analytic solution
    local_rho = sources.rho.trim()
    local_rho_an = rho_an(xg, yg, t)
    diff2 += ((local_rho_an - local_rho)**2).mean()

    # Make figures
    if plot:
        if (it % 10 == 0):
            global_rho = concatenate(local_rho)
            global_rho_an = concatenate(local_rho_an)
            global_B = concatenate(B.trim())
            global_Bz_an = concatenate(Bz_an(xg, yg, t))
            if comm.rank == 0:
                im1.set_data(global_rho)
                im2[0].set_ydata(global_B['z'][0, :])
                im2[1].set_ydata(global_Bz_an[0, :])
                im4.set_data(B['x'])
                im5.set_data(B['y'])
                im6.set_data(B['z'])
                im7.set_data(E['x'])
                im8.set_data(E['y'])
                im9.set_data(E['z'])
                im3[0].set_ydata(global_rho[0, :]/npc)
                im3[1].set_ydata(global_rho_an[0, :]/npc)
                im1.autoscale()
                im4.autoscale()
                im5.autoscale()
                im6.autoscale()
                im7.autoscale()
                im8.autoscale()
                im9.autoscale()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)

val = numpy.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt)
tol = 6e-5*charge*npc

# Check if test has passed
assert (val < tol), (val, tol)
