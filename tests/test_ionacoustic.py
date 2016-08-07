from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
from skeletor import Ohm
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_ionacoustic(plot=False):

    # Quiet start
    quiet = True
    # Number of grid points in x- and y-direction
    nx, ny = 32, 32
    # Average number of particles per cell
    npc = 256
    # Particle charge and mass
    charge = 0.5
    mass = 1.0
    # Electron temperature
    Te = 1.0
    # Dimensionless amplitude of perturbation
    A = 0.001
    # Wavenumbers
    ikx = 1
    iky = 1
    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 0.0, 0.0
    # CFL number
    cfl = 0.5
    # Number of periods to run for
    nperiods = 1

    # x- and y-grid
    xg, yg = numpy.meshgrid(numpy.arange(nx), numpy.arange(ny))

    # Sound speed
    cs = numpy.sqrt(Te/mass)

    # Time step
    dt = cfl/cs

    # Total number of particles in simulation
    np = npc*nx*ny

    # Wave vector and its modulus
    kx = 2*numpy.pi*ikx/nx
    ky = 2*numpy.pi*iky/ny
    k = numpy.sqrt(kx*kx + ky*ky)

    # Frequency
    omega = k*cs

    # Simulation time
    tend = 2*numpy.pi*nperiods/omega

    # Number of time steps
    nt = int(tend/dt)

    def rho_an(x, y, t):
        """Analytic density as function of x, y and t"""
        return charge*(1 + A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t))

    def ux_an(x, y, t):
        """Analytic x-velocity as function of x, y and t"""
        return -omega/k*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*kx/k

    def uy_an(x, y, t):
        """Analytic y-velocity as function of x, y and t"""
        return -omega/k*A*numpy.sin(kx*x+ky*y)*numpy.cos(omega*t)*ky/k

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

    # Add thermal velocity
    vx += vtx*numpy.random.normal(size=np).astype(Float)
    vy += vty*numpy.random.normal(size=np).astype(Float)

    # Start parallel processing
    idproc, nvp = cppinit(comm)

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    grid = Grid(nx, ny, comm)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/nvp)

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

    # Initialize sources
    sources = Sources(grid, comm, dtype=Float)

    # Initialize Ohm's law solver
    ohm = Ohm(grid, temperature=Te, charge=charge)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(ions)
    # Adjust density (we should do this somewhere else)
    sources.rho /= npc
    assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
    sources.rho.add_guards()
    sources.rho.copy_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

    # Calculate electric field (Solve Ohm's law)
    ohm(sources.rho, E, destroy_input=False)
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
            vmin, vmax = charge*(1 - A), charge*(1 + A)
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
        ions.push(E, dt)

        # Update time
        t += dt

        # Deposit sources
        sources.deposit_ppic2(ions)
        # Adjust density (TODO: we should do this somewhere else)
        sources.rho /= npc
        assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
        # Boundary calls
        sources.rho.add_guards_ppic2()
        sources.rho.copy_guards_ppic2()

        assert numpy.isclose(comm.allreduce(
            sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

        # Calculate forces (Solve Ohm's law)
        ohm(sources.rho, E, destroy_input=False)
        # Set boundary condition
        E.copy_guards()

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
        tol = 1e-4*charge
        assert numpy.max(numpy.abs(rho_an(xg, yg, t) - global_rho)) < tol

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_ionacoustic(plot=args.plot)
