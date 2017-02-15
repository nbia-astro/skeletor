from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
from skeletor import Ohm, InitialCondition, Experiment
from skeletor import IO
from skeletor.manifolds.ppic2 import Manifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# Set a folder for data output (relative to the file)
data_folder = '../data/run_oct1/'
# Set a tag which can be used to quickly find groups of simulations that we
# would like to compare
tag = 'test_oct1'

# Make sure this works even when the script is not in current directory
from os import path
# Absolute path of this file
this_file_path = path.dirname(path.realpath(__file__))
data_folder = this_file_path + '/' + data_folder

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

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, nlbx=1, nubx=2, nlby=1, nuby=1)

# x- and y-grid
xg, yg = numpy.meshgrid(manifold.x, manifold.y)

# Total number of particles in simulation
np = npc*nx*ny
# Maximum number of electrons in each partition
npmax = int(1.5*np/nvp)

# Initalize IO
io = IO(data_folder, locals(), __file__, tag)

# Set the rate at which we want to dump fields
io.set_outputrate(100*dt)

# Create particle array
ions = Particles(manifold, npmax, charge=charge, mass=mass)

# Create a uniform density field
init = InitialCondition(npc, quiet=True)
init(manifold, ions)

# Perturbation to particle velocities
ions['vx'] = ux_an(ions['x'], ions['y'], t=dt/2)
ions['vy'] = uy_an(ions['x'], ions['y'], t=dt/2)

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge)

# Initialize experiment
e = Experiment(manifold, ions, ohm, io)

# Deposit charges and calculate initial electric field
e.prepare()

###########################################################################
# Run experiment
###########################################################################

e.run(dt, nt)

# Difference between numerical and analytic solution
local_rho = e.sources.rho.trim()
local_rho_an = rho_an(xg, yg, e.t)
diff2 = ((local_rho_an - local_rho)**2).mean()

# Check if test has passed
assert numpy.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt) < 4e-5*charge*npc
