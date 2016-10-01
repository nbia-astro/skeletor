from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
from skeletor import Ohm, uniform_density, Experiment
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 32, 32
# Average number of particles per cell
npc = 256
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 1.0
# Dimensionless amplitude of perturbation
A = 0.001
# Wavenumbers
ikx = 1
iky = 1
# Thermal velocity of electrons in x- and y-direction
vtx, vty = 0.0, 0.0
# CFL number
cfl = 0.5
# Number of periods to run for
nperiods = 1

# Sound speed
cs = numpy.sqrt(Te/mass)

# Time step
dt = cfl/cs

# Total number of particles in simulation
np = npc*nx*ny

# Wave vector and its modulus
kx = 2*numpy.pi*ikx/nx
ky = 2*numpy.pi*iky/ny
k = numpy.sqrt(kx*kx + ky*ky)

# Frequency
omega = k*cs

# Simulation time
tend = 2*numpy.pi*nperiods/omega

# Number of time steps
nt = int(tend/dt)

def rho_an(x, y, t):
    """Analytic density as function of x, y and t"""
    return npc*charge*(1 + A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t))

def ux_an(x, y, t):
    """Analytic x-velocity as function of x, y and t"""
    return -omega/k*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*kx/k

def uy_an(x, y, t):
    """Analytic y-velocity as function of x, y and t"""
    return -omega/k*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*ky/k

# Uniform distribution of particle positions
x, y = uniform_density(nx, ny, npc, quiet=quiet)

# Perturbation to particle velocities
vx = ux_an(x, y, t=0)
vy = uy_an(x, y, t=0)

# Add thermal velocity
vx += vtx*numpy.random.normal(size=np).astype(Float)
vy += vty*numpy.random.normal(size=np).astype(Float)

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
grid = Grid(nx, ny, comm)

# x- and y-grid
xg, yg = numpy.meshgrid(grid.x, grid.y)

# Maximum number of electrons in each partition
npmax = int(1.5*np/nvp)

# Create particle array
ions = Particles(npmax, charge, mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, grid)

# Initialize Ohm's law solver
ohm = Ohm(grid, npc, temperature=Te, charge=charge)

# Initialize experiment
e = Experiment(grid, ions, ohm)

# Deposit charges and calculate initial electric field
e.prepare()

diff2 = 0
# ##########################################################################
# # Main loop over time                                                    #
# ##########################################################################
for it in range(nt):

    e.step(dt)

    # Difference between numerical and analytic solution
    local_rho = e.sources.rho.trim()
    local_rho_an = rho_an(xg, yg, e.t)
    diff2 += ((local_rho_an - local_rho)**2).mean()

# Check if test has passed
assert numpy.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt) < 4e-5*charge*npc
