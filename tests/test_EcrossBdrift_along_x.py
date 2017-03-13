from skeletor import Float3, Field, Particles
from skeletor.manifolds.mpifft4py import Manifold
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

    # Particle charge and mass
    charge = 1
    mass = 1

    # Magnetic field in z-direction
    bz = 1

    # Electric field in y-direction
    Ey = 1/32

    # Drift velocity in x-direction
    vdx = bz*Ey/bz**2

    # Gyration frequency
    og = bz*charge/mass

    # Correct for time discretization error
    # og = ? Add later

    # Phase
    phi = 0

    # Amplitude of perturbation
    ampl = 8/32

    # Number of grid points in x- and y-direction
    nx, ny = 32, 64

    # Total number of particles in simulation
    np = 1

    x0 = 8/32
    y0 = 33/32

    x0 = numpy.array(x0)
    y0 = numpy.array(y0)

    def x_an(t): return -ampl*numpy.cos(og*t + phi) + x0 + vdx*t

    def y_an(t): return +ampl*numpy.sin(og*t + phi) + y0

    def vx_an(t): return (vdx+og*ampl*numpy.sin(og*t + phi))*numpy.ones(np)

    def vy_an(t): return og*ampl*numpy.cos(og*t + phi)*numpy.ones(np)

    # Particle position at t = -dt/2
    x = x_an(-dt/2)
    y = y_an(-dt/2)

    # Particle velocity at t = 0
    vx = vx_an(t=0)
    vy = vy_an(t=0)
    vz = numpy.zeros_like(vx)

    # Drift forward by dt/2
    x += vx*dt/2
    y += vy*dt/2

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm, Lx=1.0, Ly=2.0)

    # x- and y-grid
    xg, yg = numpy.meshgrid(manifold.x, manifold.y)

    # Maximum number of ions in each partition
    # For this test we only have one particle.
    npmax = np

    # Create particle array
    ions = Particles(manifold, npmax, charge=charge, mass=mass)

    # Assign particles to subdomains
    ions.initialize(x, y, vx, vy, vz)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Set the electric field to zero
    E = Field(manifold, dtype=Float3)
    E.fill((0.0, 0.0, 0.0))
    E['y'] = Ey
    E.copy_guards()

    B = Field(manifold, dtype=Float3)
    B.fill((0.0, 0.0, bz))
    B.copy_guards()

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings

        plt.rc('image', origin='lower', interpolation='nearest')
        plt.figure(1)
        fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
        x = ions['x'][0]*manifold.dx
        y = ions['y'][0]*manifold.dy
        lines = ax1.plot(x, y, 'b.', x_an(0), y_an(0), 'rx')
        ax1.set_xlim(-manifold.dx, manifold.Lx + manifold.dx)
        ax1.set_ylim(-manifold.dy, manifold.Ly + manifold.dy)

    t = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        ions.push(E, B, dt)

        assert comm.allreduce(ions.np, op=MPI.SUM) == np

        # Update time
        t += dt
        # True if particle is in this domain
        ind = numpy.logical_and(ions['y'][0] >= manifold.edges[0],
                                ions['y'][0] < manifold.edges[1])
        if ind:
            diff_x = abs(ions['x'][0]*manifold.dx - x_an(t))
            diff_y = abs(ions['y'][0]*manifold.dy - y_an(t))
            err = numpy.max([diff_x, diff_y])/ampl
            # Round off errrors giving trouble when comparing
            if err > 1.0:
                err = 0.0
            # print(err, ions['y'][0], y_an(t), ions['x'][0], x_an(t))
            # Check if test has passed
            assert(err < 1.0e-3)

        # Make figures
        if plot:
            if (it % 300 == 0):
                if ind:
                    x = ions['x'][0]*manifold.dx
                    y = ions['y'][0]*manifold.dy
                    lines[0].set_data(x, y)
                    lines[1].set_data(x_an(t), y_an(t))
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                    "ignore", category=mplDeprecation)
                        plt.pause(1e-7)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_EcrossBdrift(plot=args.plot)
