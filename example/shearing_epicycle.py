from skeletor import cppinit, Float3, Field, Particles
import numpy
from skeletor.manifolds.second_order import ShearingManifold
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

plot = True
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

# Modified magnetic field
bz_star = bz + 2.0*mass/charge*Omega

# Spin
Sz = ocbz + 2.0*Omega

# Gyration frequency
og = numpy.sqrt(Sz*(Sz + S))

# Phase
phi = numpy.pi/2

# Number of grid points in x- and y-direction
nx, ny = 64, 32

# Grid size
Lx = 2
Ly = 1

# Amplitude of perturbation
ampl = Lx/3

# Total number of particles in simulation
N = 1

y0 = Ly/2
x0 = Lx/2

x0 = numpy.array(x0)
y0 = numpy.array(y0)


def y_an(t):
    return ampl*numpy.cos(og*t + phi)*numpy.ones(N) + y0


def x_an(t):
    x = +(Sz/og)*ampl*numpy.sin(og*t+phi)*numpy.ones(N) + x0 - S*t*y0
    return x


def vy_an(t):
    return -og*ampl*numpy.sin(og*t + phi)*numpy.ones(N)


def vx_an(t):
    return (Sz*ampl*numpy.cos(og*t + phi) - S*y0)*numpy.ones(N)


# Particle position at t = -dt/2
x = x_an(-dt/2)
y = y_an(-dt/2)

# Particle velocity at t = 0
vx = vx_an(t=0)
vy = vy_an(t=0)
vz = numpy.zeros_like(vy)

# Drift forward by dt/2
x += vx*dt/2
y += vy*dt/2


# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = ShearingManifold(nx, ny, comm, S=S, Omega=Omega, Lx=Lx, Ly=Ly)

# x- and y-grid
xg, yg = numpy.meshgrid(manifold.x, manifold.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
Nmax = N

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, vz)

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.N, op=MPI.SUM) == N

E = Field(manifold, comm, dtype=Float3)
E.fill((0.0, 0.0, 0.0))

B = Field(manifold, comm, dtype=Float3)
B.fill((0.0, 0.0, 0.0))

# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings

    plt.rc('image', origin='upper', interpolation='nearest')
    plt.figure(1)
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
    lines1 = ax1.plot(ions['y'][0], ions['y'][0], 'b.',)
    lines2 = ax2.plot(ions['vx'][0], ions['vy'][0], 'b.')
    ax1.set_xlim(-1, ny+1)
    ax1.set_ylim(-1, nx+1)
    ax2.set_ylim(-1.1*og*ampl, 1.1*og*ampl)
    ax2.set_xlim(-Sz*ampl - S*y0, Sz*ampl - S*y0)
    ax1.set_xlabel('y')
    ax1.set_ylabel('x')
    ax2.set_xlabel('vx')
    ax2.set_ylabel('vy')

t = 0
##########################################################################
# Main loop over time                                                    #
##########################################################################
for it in range(nt):
    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    # ions.push(E_star, B, dt)
    ions.push_modified(E, B, dt)

    # True if particle is in this domain
    ind = numpy.logical_and(ions['y'][0] >= manifold.edges[0],
                            ions['y'][0] < manifold.edges[1])

    # Update time
    t += dt

    # Make figures
    if plot:
        if (it % 100 == 0):
            if ind:
                lines1[0].set_data(ions['y'][0], ions['x'][0])
                lines2[0].set_data(ions['vx'][0], ions['vy'][0])
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                    plt.pause(1e-7)

# Check if test has passed
# This test is only a visual test at the moment...
