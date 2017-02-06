from skeletor import Float, Float2, Field, Particles, Sources
from skeletor import Ohm, Faraday, DensityPertubation
from skeletor.manifolds.second_order import Manifold
from skeletor.predictor_corrector import Experiment
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
from dispersion_solvers import HallDispersion
from numpy import cos, sin, pi, arctan

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
iky = 0
# Thermal velocity of electrons in x- and y-direction
vtx, vty = 0.0, 0.0
# CFL number
cfl = 0.1
# Number of periods to run for
nperiods = 2.0

# Sound speed
cs = numpy.sqrt(Te/mass)

# Total number of particles in simulation
np = npc*nx*ny

# Wave vector and its modulus
kx = 2*numpy.pi*ikx/Lx
ky = 2*numpy.pi*iky/Ly
k = numpy.sqrt(kx*kx + ky*ky)

# Magnetic field strength
B0 = 1

# Angle of k-vector with respect to x-axis
theta = arctan(iky/ikx)

(Bx, By, Bz) = (0, 0, B0)
B2 = Bx**2 + By**2 + Bz**2

rho0 = 1.0

va = B0

vph = numpy.sqrt(cs*2 + va**2)

# Frequency
omega = k*vph

# Simulation time
tend = 2*numpy.pi*nperiods/omega

# Ohmic resistivity
eta = 0

di = HallDispersion(kperp=kx, kpar=0, va=va, cs=cs, eta=eta, along_x=False,
                    theta=theta)

# Mode number
m = 0

# Phase factor
phase = lambda x, y, t: A*numpy.exp(1j*(di.omega[m]*t - kx*x - ky*y))

# Linear solutions in real space
rho_an = lambda x, y, t: rho0 + rho0*(di.vec[m]['drho']*phase(x, y, t)).real
Bx_an = lambda x, y, t: Bx + B0*(di.vec[m]['bx']*phase(x, y, t)).real
By_an = lambda x, y, t: By + B0*(di.vec[m]['by']*phase(x, y, t)).real
Bz_an = lambda x, y, t: Bz + B0*(di.vec[m]['bz']*phase(x, y, t)).real
Ux_an = lambda x, y, t:         (di.vec[m]['vx']*phase(x, y, t)).real
Uy_an = lambda x, y, t:         (di.vec[m]['vy']*phase(x, y, t)).real
Uz_an = lambda x, y, t:         (di.vec[m]['vz']*phase(x, y, t)).real

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

# Create a uniform density field
init = DensityPertubation(npc, ikx, iky, ampl=A)
init(manifold, ions)

# Perturbation to particle velocities
ions['vx'] += Ux_an(ions['x'], ions['y'], t=0)
ions['vy'] += Uy_an(ions['x'], ions['y'], t=0)
ions['vz'] += Uz_an(ions['x'], ions['y'], t=0)

ions.from_units()

B = Field(manifold, dtype=Float2)
B['x'].active = Bx_an(xg, yg, t=0)
B['y'].active = By_an(xg, yg, t=0)
B['z'].active = Bz_an(xg, yg, t=0)
B.copy_guards()

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge, npc=npc, eta=eta)

# Initialize experiment
e = Experiment(manifold, ions, ohm, B, io=None)

# Deposit charges and calculate initial electric field
e.prepare(dt)

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
        im1 = axes[0,0].imshow(global_rho/npc, vmin=vmin, vmax=vmax)
        im2 = axes[0,1].plot(xg[0, :], global_B['z'][0, :], 'b',
                             xg[0, :], global_Bz_an[0, :], 'k--')
        im3 = axes[0,2].plot(xg[0, :], global_rho[0, :]/npc, 'b',
                       xg[0, :], global_rho_an[0, :], 'k--')
        im4 = axes[1,0].imshow(global_B['x'])
        im5 = axes[1,1].imshow(global_B['y'])
        im6 = axes[1,2].imshow(global_B['z'], vmin=vmin, vmax=vmax)
        im7 = axes[2,0].imshow(e.E['x'], vmin=-A, vmax=A)
        im8 = axes[2,1].imshow(e.E['y'], vmin=-A, vmax=A)
        im9 = axes[2,2].imshow(e.E['z'])
        axes[0,0].set_title(r'$\rho$')
        axes[0,1].set_title(r'$B_z$')
        axes[0,2].set_title(r'$\rho$')
        axes[1,0].set_title(r'$B_x$')
        axes[1,1].set_title(r'$B_y$')
        axes[1,2].set_title(r'$B_z$')
        axes[2,0].set_title(r'$E_x$')
        axes[2,1].set_title(r'$E_y$')
        axes[2,2].set_title(r'$E_z$')
        axes[0,2].set_ylim(vmin, vmax)
        axes[0,1].set_ylim(vmin, vmax)
        axes[0,2].set_xlim(0, Lx)

diff2 = 0
##########################################################################
# Main loop over time                                                    #
##########################################################################
for it in range(nt):

    # The update is handled by the experiment class
    e.iterate(dt)

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
            global_Bz_an = concatenate(Bz_an(xg+manifold.dx/2, yg, e.t))
            if comm.rank == 0:
                im1.set_data(global_rho/npc)
                im2[0].set_ydata(global_B['z'][0, :])
                im2[1].set_ydata(global_Bz_an[0, :])
                im4.set_data(global_B['x'])
                im5.set_data(global_B['y'])
                im6.set_data(global_B['z'])
                im7.set_data(e.E['x'])
                im8.set_data(e.E['y'])
                im9.set_data(e.E['z'])
                im3[0].set_ydata(global_rho[0, :]/npc)
                im3[1].set_ydata(global_rho_an[0, :])
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)

val = numpy.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt)
tol = 6e-5*charge*npc

# Check if test has passed
assert (val < tol), (val, tol)