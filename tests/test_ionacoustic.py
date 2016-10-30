from skeletor import Float, Float2, Grid, Field, Particles, Sources
from skeletor import Ohm
from skeletor.operators.mpifft4py import Operators
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_ionacoustic(plot=False):

    # Order of interpolation used for deposition and push
    order = 'tsc'
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
        return npc*charge*(1 + A*numpy.cos(kx*x+ky*y)*numpy.sin(omega*t))

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
                numpy.arange(dx/2, nx+dx/2, dx),
                numpy.arange(dx/2, ny+dx/2, dy))
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

    x += vx*dt/2
    y += vy*dt/2

    x = numpy.mod(x, nx)
    y = numpy.mod(y, ny)

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    if order == 'tsc':
        grid = Grid(nx, ny, comm, nlbx=1, nubx=1, nlby=1, nuby=1)
    else:
        grid = Grid(nx, ny, comm, nlbx=0, nubx=2, nlby=0, nuby=1)

    ax = 0
    ay = 0
    operators = Operators(grid, ax, ay, np)
    grid.operators = operators

    # x- and y-grid
    xg, yg = numpy.meshgrid(grid.x, grid.y)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/comm.size)

    # Create particle array
    ions = Particles(npmax, charge, mass, order=order)

    # Assign particles to subdomains
    ions.initialize(x, y, vx, vy, grid)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Set the electric field to zero
    E = Field(grid, dtype=Float2)
    E.fill((0.0, 0.0))

    # Initialize sources
    sources = Sources(grid, order=order, dtype=Float)

    # Initialize Ohm's law solver
    ohm = Ohm(grid, npc, temperature=Te, charge=charge)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(ions)
    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)

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
        global_rho_an = concatenate(rho_an(xg, yg, 0))

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            plt.clf()
            fig, (ax1, ax2, ax3) = plt.subplots(num=1, ncols=3)
            vmin, vmax = npc*charge*(1 - A), npc*charge*(1 + A)
            im1 = ax1.imshow(global_rho, vmin=vmin, vmax=vmax)
            im2 = ax2.imshow(global_rho_an, vmin=vmin, vmax=vmax)
            im3 = ax3.plot(xg[0, :], global_rho[0, :], 'b',
                           xg[0, :], global_rho_an[0, :], 'k--')
            ax1.set_title(r'$\rho$')
            ax3.set_ylim(vmin, vmax)
            ax3.set_xlim(0, x[-1])

    t = 0
    diff2 = 0
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
        sources.deposit(ions)

        # Boundary calls
        sources.rho.add_guards()

        # Calculate forces (Solve Ohm's law)
        ohm(sources.rho, E, destroy_input=False)
        # Set boundary condition
        E.copy_guards()

        # Difference between numerical and analytic solution
        local_rho = sources.rho.trim()
        local_rho_an = rho_an(xg, yg, t)
        diff2 += ((local_rho_an - local_rho)**2).mean()

        # Make figures
        if plot:
            if (it % 1 == 0):
                global_rho = concatenate(local_rho)
                global_rho_an = concatenate(local_rho_an)
                if comm.rank == 0:
                    im1.set_data(global_rho)
                    im2.set_data(global_rho_an)
                    im3[0].set_ydata(global_rho[0, :])
                    im3[1].set_ydata(global_rho_an[0, :])
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)

    val = numpy.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt)
    tol = 6e-5*charge*npc

    # Check if test has passed
    assert (val < tol), (val, tol)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_ionacoustic(plot=args.plot)
