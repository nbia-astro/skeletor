from skeletor import Float, Float2, Field, Particles, Sources
from skeletor.manifolds.mpifft4py import Manifold
from skeletor import Poisson
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_plasmafrequency(plot=False):

    # Quiet start
    quiet = True
    # Number of grid points in x- and y-direction
    nx, ny = 32, 32
    # Average number of particles per cell
    npc = 256
    # Particle charge and mass
    charge = -1
    mass = 1.0
    # Background ion density
    n0 = 1.0
    # Dimensionless amplitude of perturbation
    A = 0.001
    # Wavenumbers
    ikx = 1
    iky = 1
    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 0.0, 0.0
    # Number of periods to run for
    nperiods = 1

    # x- and y-grid
    xg, yg = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))

    # Total number of particles in simulation
    np = npc*nx*ny

    # Epsilon 0
    eps0 = 1

    # Plasma frequency
    omega = numpy.sqrt(charge**2*n0/(mass*eps0))

    # Time step
    dt = 0.1

    # Wave vector and its modulus
    kx = 2*numpy.pi*ikx/nx
    ky = 2*numpy.pi*iky/ny
    k = numpy.sqrt(kx*kx + ky*ky)

    # Simulation time
    tend = 2*numpy.pi*nperiods/omega

    # Number of time steps
    nt = int(tend/dt)

    def rho_an(x, y, t):
        """Analytic density as function of x, y and t"""
        return npc*charge*A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t)

    def ux_an(x, y, t):
        """Analytic x-velocity as function of x, y and t"""
        return -omega*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*kx/k**2

    def uy_an(x, y, t):
        """Analytic y-velocity as function of x, y and t"""
        return -omega*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*ky/k**2

    if quiet:
        # Uniform distribution of particle positions (quiet start)
        sqrt_npc = int(numpy.sqrt(npc))
        assert sqrt_npc**2 == npc
        dx = dy = 1/sqrt_npc
        x, y = numpy.meshgrid(
                numpy.arange(0, nx, dx),
                numpy.arange(0, ny, dy))
        x = x.flatten()
        y = y.flatten()
    else:
        x = nx*numpy.random.uniform(size=np).astype(Float)
        y = ny*numpy.random.uniform(size=np).astype(Float)

    # Perturbation to particle velocities
    vx = ux_an(x, y, t=0)
    vy = uy_an(x, y, t=0)

    # x -= A/kx*numpy.sin(kx*x+ky*y)

    # Add thermal velocity
    vx += vtx*numpy.random.normal(size=np).astype(Float)
    vy += vty*numpy.random.normal(size=np).astype(Float)

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/comm.size)

    # Create particle array
    electrons = Particles(manifold, npmax, charge, mass)

    # Assign particles to subdomains
    electrons.initialize(x, y, vx, vy)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(electrons.np, op=MPI.SUM) == np

    # Set the electric field to zero
    E = Field(manifold, dtype=Float2)
    E.fill((0.0, 0.0))

    # Initialize sources
    sources = Sources(manifold, dtype=Float)

    # Initialize integro-differential operators
    poisson = Poisson(manifold, np)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(electrons)
    # Adjust density (we should do this somewhere else)
    # sources.rho /= npc
    # assert numpy.isclose(sources.rho.sum(), electrons.np*charge/npc)
    sources.rho.add_guards_ppic2()
    sources.rho += n0*npc
    # assert numpy.isclose(comm.allreduce(
    # sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

    # Solve Gauss' law
    poisson(sources.rho, E)
    # Set boundary condition
    E.copy_guards_ppic2()

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return numpy.concatenate(comm.allgather(arr))

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings
        global_rho = concatenate(sources.rho.trim())

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            fig, (ax1, ax2, ax3) = plt.subplots(num=1, ncols=3)
            vmin, vmax = npc*charge*A, -npc*charge*A
            im1 = ax1.imshow(rho_an(xg, yg, 0), vmin=vmin, vmax=vmax)
            im2 = ax2.imshow(rho_an(xg, yg, 0), vmin=vmin, vmax=vmax)
            im3 = ax3.plot(xg[0, :], global_rho[0, :], 'b',
                           xg[0, :], rho_an(xg, yg, 0)[0, :], 'k--')
            ax1.set_title(r'$\rho$')
            ax3.set_ylim(vmin, vmax)
            ax3.set_xlim(0, x[-1])

    t = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        electrons.push(E, dt)

        # Update time
        t += dt

        # Deposit sources
        sources.deposit_ppic2(electrons)
        # Adjust density (TODO: we should do this somewhere else)
        # sources.rho /= npc
        # assert numpy.isclose(sources.rho.sum(),electrons.np*charge/npc)
        # Boundary calls
        sources.rho.add_guards_ppic2()
        sources.rho += n0*npc

        # assert numpy.isclose(comm.allreduce(
        #     sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

        # Solve Gauss' law
        poisson(sources.rho, E)

        # Set boundary condition
        E.copy_guards_ppic2()

        # Make figures
        if plot:
            if (it % 1 == 0):
                global_rho = concatenate(sources.rho.trim())
                if comm.rank == 0:
                    im1.set_data(global_rho)
                    im2.set_data(rho_an(xg, yg, t))
                    im3[0].set_ydata(global_rho[0, :])
                    im3[1].set_ydata(rho_an(xg, yg, t)[0, :])
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)

    # Check if test has passed
    global_rho = concatenate(sources.rho.trim())
    if comm.rank == 0:
        tol = 1e-4*npc
        err = numpy.max(numpy.abs(rho_an(xg, yg, t) - global_rho))
        assert (err < tol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_plasmafrequency(plot=args.plot)
