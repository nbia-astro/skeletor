from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

def test_EcrossBdrift(plot=False):

    # Time step
    dt = 1e-3

    # Simulation time
    tend = 25

    # Number of time steps
    nt = int(tend/dt)

    t0 = -dt/2

    # Particle charge and mass
    charge = 1
    mass   = 1

    # Magnetic field in z-direction
    bz = 1

    # Electric field in y-direction
    Ey = 1

    # Drift velocity in x-direction
    vdx = bz*Ey/bz**2

    # Gyration frequency
    og = bz*charge/mass

    # Correct for time discretization error
    # og = ? Add later

    # Phase
    phi = 0

    # Amplitude of perturbation
    ampl = 8

    # Number of grid points in x- and y-direction
    nx, ny = 32, 64

    # Total number of particles in simulation
    np = 1

    x0 = 8.
    y0 = 32.

    x0 = numpy.array(x0)
    y0 = numpy.array(y0)

    def  x_an(t): return -ampl*numpy.cos (og*t + phi) + x0 + vdx*t
    def  y_an(t): return +ampl*numpy.sin (og*t + phi) + y0
    def vx_an(t): return (vdx+og*ampl*numpy.sin (og*t + phi))*numpy.ones(np)
    def vy_an(t): return og*ampl*numpy.cos (og*t + phi)*numpy.ones(np)


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

    # Set the electric field to zero
    E = Field(grid, comm, dtype=Float2)
    E.fill((0.0, 0.0))
    E['y'] = Ey
    E.copy_guards_ppic2()

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
            lines = ax1.plot(ions['x'], ions['y'], 'b.', x_an(0), y_an(0), 'rx')
            ax1.set_xlim(-1, nx+1)
            ax1.set_ylim(-1, ny+1)

    t = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        ions.push(E, dt, bz)

        assert comm.allreduce(ions.np, op=MPI.SUM) == np

        # Update time
        t += dt

        err = numpy.max(numpy.abs([ions['x']-x_an(t), ions['y']-y_an(t)]))/ampl

        # Make figures
        if plot:
            if (it % 300 == 0):
                if comm.rank == 0:
                    lines[0].set_data(ions['x'], ions['y'])
                    lines[1].set_data(x_an(t), y_an(t))
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                    "ignore", category=mplDeprecation)
                        plt.pause(1e-7)

    # Check if test has passed
    assert(err < 1e-3)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_EcrossBdrift(plot=args.plot)
