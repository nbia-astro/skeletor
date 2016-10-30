from skeletor import cppinit, Float, Grid, Particles, Sources
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_deposit_tsc(plot=False):

    # Quiet start
    quiet = True
    # Number of grid points in x- and y-direction
    nx, ny = 16, 16
    # Average number of particles per cell
    npc = 1
    # Particle charge and mass
    charge = 0.5
    mass = 1.0
    # Electron temperature
    Te = 1.0
    # Dimensionless amplitude of perturbation
    A = 0.5
    # Wavenumbers
    ikx = 1
    iky = 1
    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 0.0, 0.0
    # CFL number
    cfl = 0.5

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
                numpy.arange(dy/2, ny+dy/2, dy))
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

    x += dt*vx/2
    y += dt*vy/2

    x = numpy.mod(x, nx)
    y = numpy.mod(y, ny)

    # Start parallel processing
    idproc, nvp = cppinit(comm)

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    grid = Grid(nx, ny, comm, nlbx=1, nubx=2, nlby=1, nuby=2)

    # x- and y-grid
    xg, yg = numpy.meshgrid(grid.x, grid.y)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/nvp)

    # Create particle array
    ions = Particles(npmax, charge, mass)

    # Assign particles to subdomains
    ions.initialize(x, y, vx, vy, grid)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Initialize sources
    sources = Sources(grid, dtype=Float)

    # Deposit sources
    sources.deposit(ions)
    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)
    rho_cic = sources.rho.trim().copy()

    sources.deposit_tsc(ions)
    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)
    rho_tsc = sources.rho.trim().copy()

    diff2 = ((rho_tsc - rho_cic)**2).mean()

    # Difference between numerical and analytic solution
    # local_rho = sources.rho.trim()
    # local_rho_an = rho_an(xg, yg, dt/2)
    # diff2 = ((local_rho_an - local_rho)**2).mean()
    err = numpy.sqrt(comm.allreduce(diff2, op=MPI.SUM))


    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt

        # Concatenate local arrays to obtain global arrays
        # The result is available on all processors.
        def concatenate(arr):
            return numpy.concatenate(comm.allgather(arr))

        global_rho_tsc = concatenate(rho_tsc)
        global_rho_cic = concatenate(rho_cic)

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            plt.clf()
            fig, (ax1, ax2) = plt.subplots(num=1, ncols=2)
            ax1.imshow(global_rho_tsc)
            ax2.plot(global_rho_tsc[0, :], 'b',
                     global_rho_tsc[:, nx//2], 'r')
            ax2.plot(global_rho_cic[0, :], 'k--',
                     global_rho_cic[:, nx//2], 'k--')
            plt.show()

    assert(err<0.002), err

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_deposit_tsc(plot=args.plot)
