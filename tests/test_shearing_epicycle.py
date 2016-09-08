from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

# def test_shearing_epicycle(plot=False):
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
mass   = 1

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
og = numpy.sqrt (Sz*(Sz + S))

# Correct for discretization error
# og = numpy.arcsin (numpy.sqrt ((og*dt/2)**2/(1.0 + (Sz*dt/2)**2)))/(dt/2)

# Phase
phi = numpy.pi/2

# Amplitude of perturbation
ampl = 20

# Number of grid points in x- and y-direction
nx, ny = 64, 32

# Total number of particles in simulation
np = 1

y0 = 16.
x0 = 32.

x0 = numpy.array(x0)
y0 = numpy.array(y0)

def  y_an(t): return          ampl*numpy.cos (og*t + phi)*numpy.ones(np) + y0
def  x_an(t): return +(Sz/og)*ampl*numpy.sin (og*t + phi)*numpy.ones(np) + x0 - S*t*(y0-ny/2)
def vy_an(t): return -og*ampl*numpy.sin (og*t + phi)*numpy.ones(np)
def vx_an(t): return (Sz*ampl*numpy.cos (og*t + phi) - S*(y0-ny/2))*numpy.ones(np)


# Particle position at t = -dt/2
x = x_an(-dt/2)
y = y_an(-dt/2)

# Particle velocity at t = 0
vx = vx_an(t=0)
vy = vy_an(t=0)

# Drift forward by dt/2
x += vx*dt/2
y += vy*dt/2


# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
grid = Grid(nx, ny, comm)

# x- and y-grid
xg, yg = numpy.meshgrid(grid.x, grid.y)

# Grid that extends into the ghost-zones (not MPI-friendly...)
xgg, ygg = numpy.meshgrid(range(nx+2), range(ny+1))

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
npmax = int(1.25*np/nvp)

# Create particle array
ions = Particles(npmax, charge, mass)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, grid)

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.np, op=MPI.SUM) == np

# Electric field in y-direction
E_star = Field(grid, comm, dtype=Float2)
E_star.fill((0.0, 0.0))
E_star['y'] = -2*S*(ygg-ny/2)*mass/charge*Omega


# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings

    if comm.rank == 0:
        plt.rc('image', origin='upper', interpolation='nearest')
        plt.figure(1);plt.clf()
        fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
        lines1 = ax1.plot(ions['y'], ions['y'], 'b.', x_an(0), y_an(0), 'rx')
        lines2 = ax2.plot(ions['vx'], ions['vy'], 'b.',  vx_an(0), vy_an(0), 'rx')
        ax1.set_xlim(-1, ny+1)
        ax1.set_ylim(-1, nx+1)
        ax2.set_ylim(-1.1*og*ampl, 1.1*og*ampl)
        ax2.set_xlim((-Sz*ampl+S*(y0-ny/2)), (Sz*ampl+S*(y0-ny/2)))
        ax1.set_xlabel('y')
        ax1.set_ylabel('x')
        ax2.set_xlabel('vx')
        ax2.set_ylabel('vy')
        dat = numpy.load("pos.npz", encoding='bytes')
        for (r, v) in zip (dat['pos'], dat['vel']):
            x, y, z = zip (*r)
            x = numpy.array(x) + ny/2
            y = numpy.array(y) + nx/2
            ax1.plot (x, y, 'k--')
        ax1.invert_yaxis()

t = 0
##########################################################################
# Main loop over time                                                    #
##########################################################################
for it in range(nt):
    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    ions.push(E_star, dt, bz_star)

    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Update time
    t += dt

    err = numpy.max(numpy.abs([ions['x']-x_an(t), ions['y']-
                    numpy.mod(y_an(t), ny)]))/ampl

    # Make figures
    if plot:
        if (it % 100 == 0):
            if comm.rank == 0:
                lines1[0].set_data(ions['y'], ions['x'])
                lines1[1].set_data(numpy.mod(y_an(t), ny), x_an(t))
                lines2[0].set_data(ions['vx'], ions['vy'])
                lines2[1].set_data(vx_an(t), vy_an(t))
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                    plt.pause(1e-7)

# Check if test has passed
# assert(err < 2e-3)

# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--plot', '-p', action='store_true')
#     args = parser.parse_args()

#     test_shearing_epicycle(plot=args.plot)
