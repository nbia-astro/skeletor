from skeletor import Float, Float3, Field, Particles
from skeletor import Ohm, InitialCondition, State
from skeletor.manifolds.second_order import Manifold
from skeletor.time_steppers.horowitz import TimeStepper
import numpy as np
from mpi4py import MPI
from dispersion_solvers import HallDispersion
from writer import write_grid, write_time, write_fields, write_particles
import h5py
import subprocess
comm = MPI.COMM_WORLD

# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 16, 32
# Grid size in x- and y-direction (square cells!)
Lx = 16
Ly = 16
dx = Lx/nx
# Average number of particles per cell
npc = 1
# Particle charge and mass
charge = 0.001
mass = 1.0
# Electron temperature
Te = 0.0
# Dimensionless amplitude of perturbation
A = 0.005
# Wavenumbers
ikx = 1
iky = 0
# Number of periods to run for
nperiods = 1.0

# Sound speed
cs = np.sqrt(Te/mass)

# Total number of particles in simulation
N = npc*nx*ny

# Wave vector and its modulus
kx = 2*np.pi*ikx/Lx
ky = 2*np.pi*iky/Ly
k = np.sqrt(kx*kx + ky*ky)

# Angle of k-vector with respect to x-axis
theta = np.arctan(iky/ikx) if ikx != 0 else np.pi/2

# Magnetic field strength
B0 = 1

(Bx, By, Bz) = (B0*np.cos(theta), B0*np.sin(theta), 0)

rho0 = 1.0

va = B0

# Ohmic resistivity
eta = 0

# Cyclotron frequency
oc = charge*B0/mass

# Hall parameter
etaH = va**2/oc

di = HallDispersion(kperp=0, kpar=k, va=va, cs=cs, etaH=etaH, eta=eta,
                    along_x=True, theta=theta)

# Mode number
m = 0

omega = di.omega[m].real


def frequency(kzva):
    hel = 1
    return kzva*(np.sqrt(1.0 + (0.5*kzva/oc)**2) + 0.5*kzva/(hel*oc))


def get_dt(kzva):
    dt = 1/frequency(kzva)
    dt = 2.0**(np.floor(np.log2(dt)))
    return dt


kmax = np.pi/dx

# Simulation time
tend = 2*np.pi*nperiods/omega


# Phase factor
def phase(x, y, t):
    return A*np.exp(1j*(di.omega[m]*t - kx*x - ky*y))


# Linear solutions in real space
def rho_an(x, y, t):
    return rho0 + rho0*(di.vec[m]['drho']*phase(x, y, t)).real


def Bx_an(x, y, t):
    return Bx + B0*(di.vec[m]['bx']*phase(x, y, t)).real


def By_an(x, y, t):
    return By + B0*(di.vec[m]['by']*phase(x, y, t)).real


def Bz_an(x, y, t):
    return Bz + B0*(di.vec[m]['bz']*phase(x, y, t)).real


def Ux_an(x, y, t):
    return (di.vec[m]['vx']*phase(x, y, t)).real


def Uy_an(x, y, t):
    return (di.vec[m]['vy']*phase(x, y, t)).real


def Uz_an(x, y, t):
    return (di.vec[m]['vz']*phase(x, y, t)).real


# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly)

# Time step
# dt = cfl*manifold.dx/vph
dt = get_dt(kmax*va)

# Number of time steps
nt = int(tend/dt)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass)

# Create a uniform density field
init = InitialCondition(npc, quiet=quiet)
init(manifold, ions)

# Particle position in physical units
x = ions['x']*manifold.dx
y = ions['y']*manifold.dy

# Perturbation to particle velocities
ions['vx'] = Ux_an(x, y, t=0)
ions['vy'] = Uy_an(x, y, t=0)
ions['vz'] = Uz_an(x, y, t=0)


def B_an(t):
    B_an = Field(manifold, dtype=Float3)
    B_an['x'].active = Bx_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
    B_an['y'].active = By_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
    B_an['z'].active = Bz_an(xg+manifold.dx/2, yg+manifold.dy/2, t=t)
    return B_an


# Create vector potential
A_an = Field(manifold, dtype=Float3)
A_an['x'].active = 0
A_an['y'].active = -((Bz + B0*(di.vec[m]['bz']*phase(xg, yg, 0))) /
                     (1j*kx)).real
A_an['z'].active = ((By + B0*(di.vec[m]['by']*phase(xg, yg, 0))) /
                    (1j*kx)).real
A_an.copy_guards()

# Set initial magnetic field perturbation using the vector potential
B = Field(manifold, dtype=Float3)
manifold.curl(A_an, B, down=False)
# Add background magnetic field
B['x'].active += Bx
B['y'].active += By
B['z'].active += Bz
B.copy_guards()

div = Field(manifold, dtype=Float)
div.fill(0)
manifold.divergence(B, div)

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge, eta=eta)

# Initialize state
state = State(ions, B)

# Initialize timestepper
e = TimeStepper(state, ohm, manifold)

# Deposit charges and calculate initial electric field
e.prepare(dt)


diff2 = 0
##########################################################################
# Main loop over time                                                    #
##########################################################################
snap = 0
subprocess.call('mkdir data', shell=True)
subprocess.call('mkdir data/id{}'.format(comm.rank), shell=True)

for it in range(nt):

    # The update is handled by the experiment class
    e.iterate(dt)

    if it % 100 == 0:
        if comm.rank == 0:
            print(e.t, it, dt)

        filename = 'data/id{}/fields{:d}.h5'.format(comm.rank, snap)
        file = h5py.File(filename, 'w')
        write_grid(file, manifold, write_ghosts=True)
        write_time(file, e.t, it)
        write_fields(file, e.E, e.B, e.sources, write_ghosts=True)
        file.close()

        filename = 'data/id{}/particles{:d}.h5'.format(comm.rank, snap)
        file = h5py.File(filename, 'w')
        write_particles(file, ions)
        write_time(file, e.t, it)
        file.close()

        # Update snap shot number on all processors
        snap += 1
