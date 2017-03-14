from skeletor import Float3, Field, Particles
from skeletor.manifolds.mpifft4py import ShearingManifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_shearing_epicycle(plot=False):

    # Time step
    dt = 0.5e-3

    # Simulation time
    tend = 20

    # Number of time steps
    nt = int(tend/dt)

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

    # Spin
    Sz = ocbz + 2.0*Omega

    # Gyration frequency
    og = numpy.sqrt(Sz*(Sz + S))

    # Correct for discretization error
    # og = numpy.arcsin (numpy.sqrt ((og*dt/2)**2/(1.0 + (Sz*dt/2)**2)))/(dt/2)

    # Phase
    phi = 0

    # Amplitude of perturbation
    ampl = 0.2

    # Number of grid points in x- and y-direction
    nx, ny = 32, 64

    Lx, Ly = 1, 2

    # Total number of particles in simulation
    np = 1

    x0 = Lx/2
    y0 = Ly/2

    x0 = numpy.array(x0)
    y0 = numpy.array(y0)

    def x_an(t): return ampl*numpy.cos(og*t + phi)*numpy.ones(np) + x0

    def y_an(t):
        f = -(Sz/og)*ampl*numpy.sin(og*t + phi)*numpy.ones(np) \
             + y0 + S*t*(x0-Lx/2)
        return f

    def vx_an(t): return -og*ampl*numpy.sin(og*t + phi)*numpy.ones(np)

    def vy_an(t):
        return (-Sz*ampl*numpy.cos(og*t + phi) + S*(x0-Lx/2))*numpy.ones(np)

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
    manifold = ShearingManifold(nx, ny, comm, Lx=Lx, Ly=Ly, Omega=Omega, S=S)

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

    # Electric field in x-direction
    E_star = Field(manifold, dtype=Float3)
    E_star.fill((0.0, 0.0, 0.0))
    E_star['x'].active = -2*S*(xg-Lx/2)*mass/charge*Omega
    # Note: this does not set the correct boundary condition in x. And that's
    # fine as long as the particle doesn't cross the x-boundaries.
    E_star.copy_guards()

    B = Field(manifold, dtype=Float3)
    B.fill((0.0, 0.0, 2*mass*Omega/charge))
    B.copy_guards()

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings

        plt.rc('image', origin='lower', interpolation='nearest')
        plt.figure(1);plt.clf()
        fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
        lines1 = ax1.plot(ions['x'][0]*manifold.dx, ions['y'][0]*manifold.dy,
                          'b.', x_an(0), y_an(0), 'rx')
        lines2 = ax2.plot(ions['vx'][0], ions['vy'][0], 'b.',  vx_an(0), vy_an(0), 'rx')
        ax1.set_xlim(0, Lx)
        ax1.set_ylim(0, Ly)
        ax2.set_xlim(-1.1*og*ampl, 1.1*og*ampl)
        ax2.set_ylim(-Sz*ampl + (x0-Lx/2), Sz*ampl + (x0-Lx/2))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_xlabel('vx')
        ax2.set_ylabel('vy')

    t = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        ions.push(E_star, B, dt)

        assert comm.allreduce(ions.np, op=MPI.SUM) == np

        # Update time
        t += dt

        # True if particle is in this domain
        ind = numpy.logical_and(ions['y'][0] >= manifold.edges[0],
                                ions['y'][0] < manifold.edges[1])
        if ind:
            diff_x = abs(ions['x'][0]-x_an(t))
            diff_y = abs(ions['y'][0]-y_an(t))
            # Round off errrors giving trouble when comparing
            if diff_x > nx/2:
                diff_x = 0
            if diff_y > ny/2:
                diff_y = 0

            err = numpy.max([diff_x, diff_y])/ampl
            # Round off errrors giving trouble when comparing
            if err > 1.0:
                err = 0.0

            # print(err, ions['y'][0], y_an(t), ions['x'][0], x_an(t))
            # Check if test has passed
            # print(err, ions['y'][0], y_an(t), ions['x'][0], x_an(t))
            assert(err < 5.0e-2), 'err'

        # Make figures
        if plot:
            if (it % 200 == 0):
                if comm.rank == 0:
                    lines1[0].set_data(ions['x'][0]*manifold.dx,
                                       ions['y'][0]*manifold.dy)
                    lines1[1].set_data(x_an(t), numpy.mod(y_an(t), ny))
                    lines2[0].set_data(ions['vx'][0], ions['vy'][0])
                    lines2[1].set_data(vx_an(t), vy_an(t))
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                    "ignore", category=mplDeprecation)
                        plt.pause(1e-7)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_shearing_epicycle(plot=args.plot)
