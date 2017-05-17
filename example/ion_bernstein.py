from skeletor import Float3, Field, Particles
from skeletor import Ohm, Faraday, State, InitialCondition
from skeletor.manifolds.second_order import Manifold
from skeletor.time_steppers.predictor_corrector import TimeStepper
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

order = 2

fitplot = True
plot = False
# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 512, 1
# Grid size in x- and y-direction (square cells!)
Lx = 63.6
Ly = Lx*ny/nx

# Grid distances
dx, dy = Lx/nx, Ly/ny

# Average number of particles per cell
npc = 256
# Particle charge and mass
charge = 1.0
mass = 1.0
# Electron temperature
Te = 0.05

# Magnetic field strength
B0 = 1

# Thermal velocity of ions
beta = 1e-1
vt = np.sqrt(beta/2)

# Sound speed
cs = np.sqrt(Te/mass)

# Total number of particles in simulation
N = npc*nx*ny

(Bx, By, Bz) = (0, 0, B0)

# Simulation time
tend = 600

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly, lbx=order, lby=order)

# Time step
dt = 0.5e-3*63.6

# Number of time steps
nt = int(tend/dt)

faraday = Faraday(manifold)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass, order=order)

# Initialize ions
init = InitialCondition(npc, quiet=quiet, vt=vt)
init(manifold, ions)

# Add background magnetic field
B = Field(manifold, dtype=Float3)
B.fill((Bx, By, Bz))
B.copy_guards()


# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge)

# Initialize state
state = State(ions, B)

# Initialize timestepper
e = TimeStepper(state, ohm, manifold)

# Deposit charges and calculate initial electric field
e.prepare(dt)


# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return np.concatenate(comm.allgather(arr))


sampling_rate = 1
Nt = int(nt/sampling_rate)
Dt = dt*sampling_rate

# Big array for storing density at every time step
if comm.rank == 0:
    data = np.zeros((nx, Nt))

##########################################################################
# Main loop over time                                                    #
##########################################################################
It = 0
for it in range(nt):

    if (it % sampling_rate == 0):
        global_B = concatenate(e.B.trim())
        global_E = concatenate(e.E.trim())
        if comm.rank == 0:
            data[:, It] = global_E['x']
            It += 1
            print(e.t)

    e.iterate(dt)


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
    w = 2*np.pi*np.fft.rfftfreq(Nt)/Dt

    # Compute spacetime spectrum. Only show positive half of both frequency and
    # wavenumber spectra.
    spec = np.fft.rfft2(data.T)[:Nt//2, :]

    np.savez('ion_bernstein.npz', data=data.T, nt=Nt, w=w, kx=kx)

    dx = manifold.dx

    plt.rc('image', aspect='auto', interpolation='nearest', origin='lower')

    plt.figure(2)
    plt.clf()
    ikmin = 4
    plt.imshow((np.log(np.abs(spec[:, ikmin:])**2.)),
               extent=(kx[ikmin], kx[-1], w[0], w[-1]))
    plt.xlabel(r'$k_x$')
    plt.ylabel(r'$\omega$')
    plt.show()
