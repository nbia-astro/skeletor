from skeletor import Float, Float3, Field, Particles, Sources
from skeletor.manifolds.mpifft4py import Manifold
from skeletor import Poisson, InitialCondition
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
    # Mean electron number density
    # TODO: Pass this to the sources class. Right now the normalization of
    # charge and current density is such that the mean number density is always
    # equal to 1
    n0 = 1.0
    # Dimensionless amplitude of perturbation
    A = 0.001
    # Wavenumbers
    ikx = 1
    iky = 1
    # Number of periods to run for
    nperiods = 1

    # Total number of particles in simulation
    np = npc*nx*ny

    # Epsilon 0
    eps0 = 1

    # Plasma frequency
    omega = numpy.sqrt(charge**2*n0/(mass*eps0))

    # Time step
    dt = 0.1

    # Simulation time
    tend = 2*numpy.pi*nperiods/omega

    # Number of time steps
    nt = int(tend/dt)

    def rho_an(x, y, t):
        """Analytic density as function of x, y and t"""
        return charge*(n0 + A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t))

    def ux_an(x, y, t):
        """Analytic x-velocity as function of x, y and t"""
        return -omega*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*kx/k**2

    def uy_an(x, y, t):
        """Analytic y-velocity as function of x, y and t"""
        return -omega*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*ky/k**2

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm)

    # Wave vector and its modulus
    kx = 2*numpy.pi*ikx/manifold.Lx
    ky = 2*numpy.pi*iky/manifold.Ly
    k = numpy.sqrt(kx*kx + ky*ky)

        # x- and y-grid
    xg, yg = numpy.meshgrid(manifold.x, manifold.y)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/comm.size)

    # Create particle array
    electrons = Particles(manifold, npmax, charge=charge, mass=mass)

    # Create a uniform density field
    init = InitialCondition(npc, quiet=quiet)
    init(manifold, electrons)

    # Perturbation to particle velocities
    electrons['vx'] = ux_an(electrons['x'], electrons['y'], t=0)
    electrons['vy'] = uy_an(electrons['x'], electrons['y'], t=0)

    electrons.from_units()

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(electrons.np, op=MPI.SUM) == np

    # Set the electric field to zero
    E = Field(manifold, dtype=Float3)
    E.fill((0.0, 0.0, 0.0))

    B = Field(manifold, dtype=Float3)
    B.fill((0.0, 0.0, 0.0))
    B.copy_guards()

    # Initialize sources
    sources = Sources(manifold)

    # Initialize integro-differential operators
    poisson = Poisson(manifold)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(electrons)
    # Adjust density (we should do this somewhere else)
    # sources.rho /= npc
    # assert numpy.isclose(sources.rho.sum(), electrons.np*charge/npc)
    sources.current.add_guards()
    # assert numpy.isclose(comm.allreduce(
    # sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

    # Solve Gauss' law
    poisson(sources.rho, E)
    # Set boundary condition
    E.copy_guards()

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
            vmin, vmax = charge*(n0 + A), charge*(n0 - A)
            im1 = ax1.imshow(rho_an(xg, yg, 0), vmin=vmin, vmax=vmax)
            im2 = ax2.imshow(rho_an(xg, yg, 0), vmin=vmin, vmax=vmax)
            im3 = ax3.plot(xg[0, :], global_rho[0, :], 'b',
                           xg[0, :], rho_an(xg, yg, 0)[0, :], 'k--')
            ax1.set_title(r'$\rho$')
            ax3.set_ylim(vmin, vmax)
            ax3.set_xlim(0, manifold.Lx)

    t = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        electrons.push(E, B, dt)

        # Update time
        t += dt

        # Deposit sources
        sources.deposit(electrons)
        # Adjust density (TODO: we should do this somewhere else)
        # sources.rho /= npc
        # assert numpy.isclose(sources.rho.sum(),electrons.np*charge/npc)
        # Boundary calls
        sources.current.add_guards()

        # assert numpy.isclose(comm.allreduce(
        #     sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

        # Solve Gauss' law
        poisson(sources.rho, E)

        # Set boundary condition
        E.copy_guards()

        # Make figures
        if plot:
            if (it % 1 == 0):
                global_rho = concatenate(sources.rho.trim())
                global_rho_an = concatenate(rho_an(xg, yg, t))
                if comm.rank == 0:
                    im1.set_data(global_rho)
                    im2.set_data(global_rho_an)
                    im3[0].set_ydata(global_rho[0, :])
                    im3[1].set_ydata(global_rho_an[0, :])
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)

    # Check if test has passed
    global_rho = concatenate(sources.rho.trim())
    global_rho_an = concatenate(sources.rho.trim())
    if comm.rank == 0:
        tol = 1e-4
        err = numpy.max(numpy.abs(global_rho_an - global_rho))
        assert (err < tol)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_plasmafrequency(plot=args.plot)
