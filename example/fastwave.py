from skeletor import Float, Float2, Field, Particles, Sources
from skeletor import Ohm, Faraday, InitialCondition
from skeletor.manifolds.second_order import Manifold
from skeletor.predictor_corrector import Experiment
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

plot = True
# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 32, 32
# Grid size in x- and y-direction (square cells!)
Lx = nx
Ly = Lx*ny/nx
# Average number of particles per cell
npc = 64
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 0.0
# Dimensionless amplitude of perturbation
A = 1e-5
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

# Simulation time
tend = 2*numpy.pi*nperiods/omega

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

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, nlbx=1, nubx=1, nlby=1, nuby=1,
                    Lx=Lx, Ly=Ly)

# Time step
dt = cfl*manifold.dx/vph

# Number of time steps
nt = int(tend/dt)

faraday = Faraday(manifold)

# x- and y-grid
xg, yg = numpy.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
npmax = int(1.5*np/comm.size)

# Create particle array
ions = Particles(manifold, npmax, charge=charge, mass=mass)
ions2 = Particles(manifold, npmax, charge=charge, mass=mass)

# Create a uniform density field
init = InitialCondition(npc, quiet=quiet)
init(manifold, ions)
init(manifold, ions2)

# Perturbation to particle velocities
ions['vx'] += ux_an(ions['x'], ions['y'], t=dt/2)
ions['vy'] += uy_an(ions['x'], ions['y'], t=dt/2)
ions['vz'] += uz_an(ions['x'], ions['y'], t=dt/2)

ions.from_units()

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.np, op=MPI.SUM) == np

# Set the magnetic field to zero
B = Field(manifold, dtype=Float2)
B.fill((Bx, By, Bz))

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge, npc=npc)

# Initialize experiment
e = Experiment(manifold, ions, ions2, ohm, B, io=None)

# Deposit charges and calculate initial electric field
e.prepare()

# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return numpy.concatenate(comm.allgather(arr))

# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings

    global_rho = concatenate(e.sources.rho.trim())
    global_rho_an = concatenate(rho_an(xg, yg, 0))
    global_B = concatenate(e.B.trim())
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
        im4 = axes[1,0].imshow(global_B['x'])
        im5 = axes[1,1].imshow(global_B['y'])
        im6 = axes[1,2].imshow(global_B['z'])
        im7 = axes[2,0].imshow(e.E['x'])
        im8 = axes[2,1].imshow(e.E['y'])
        im9 = axes[2,2].imshow(e.E['z'])
        # ax1.set_title(r'$\rho$')
        axes[0,2].set_ylim(vmin, vmax)
        axes[0,1].set_ylim(vmin, vmax)
        axes[0,2].set_xlim(0, Lx)

diff2 = 0
##########################################################################
# Main loop over time                                                    #
##########################################################################
for it in range(nt):

    # The update is handled by the experiment class
    # e.step(dt, update=True)
    e.step(dt, update=True)

    # Difference between numerical and analytic solution
    local_rho = e.sources.rho.trim()
    local_rho_an = rho_an(xg, yg, e.t)
    diff2 += ((local_rho_an - local_rho)**2).mean()

    # Make figures
    if plot:
        if (it % 10 == 0):
            global_rho = concatenate(local_rho)
            global_rho_an = concatenate(local_rho_an)
            global_B = concatenate(e.B.trim())
            global_Bz_an = concatenate(Bz_an(xg, yg, e.t))
            if comm.rank == 0:
                im1.set_data(global_rho)
                im2[0].set_ydata(global_B['z'][0, :])
                im2[1].set_ydata(global_Bz_an[0, :])
                im4.set_data(global_B['x'])
                im5.set_data(global_B['y'])
                im6.set_data(global_B['z'])
                im7.set_data(e.E['x'])
                im8.set_data(e.E['y'])
                im9.set_data(e.E['z'])
                im3[0].set_ydata(global_rho[0, :]/npc)
                im3[1].set_ydata(global_rho_an[0, :]/npc)
                for im in (im1, im4, im5, im6, im7, im8, im9):
                    im.autoscale()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)

val = numpy.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt)
tol = 6e-5*charge*npc

# Check if test has passed
assert (val < tol), (val, tol)
