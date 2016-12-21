from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
from skeletor import Poisson as Poisson
# from skeletor import PoissonMpiFFT4py as Poisson
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_twostream(plot=False, fitplot=False):
    """Test of twostream instability."""

    # Number of grid points in x- and y-direction
    nx, ny = 64, 4

    # Average number of particles per cell
    npc = 2**15

    # Number of time steps
    nt = 600

    # Background ion density
    n0 = 1.0

    # Particle charge and mass
    charge = -1.0
    mass = 1.0

    # Timestep
    dt = 0.05

    # Smoothed particle size in x/y direction
    ax = 0
    ay = 0

    # Total number of particles in simulation
    np = npc*nx*ny

    # Mean velocity of electrons in x-direction
    vdx = 6.0

    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 0., 0.

    x = nx*numpy.random.uniform(size=np).astype(Float)
    y = ny*numpy.random.uniform(size=np).astype(Float)

    # Thermal component of velocity
    vx = vtx*numpy.random.normal(size=np).astype(Float)
    vy = vty*numpy.random.normal(size=np).astype(Float)

    # Make counterpropagating in x
    vx[:np//2] += vdx
    vx[np//2:] -= vdx

    # Start parallel processing
    idproc, nvp = cppinit(comm)

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    grid = Grid(nx, ny, comm)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/nvp)

    # Create particle array
    electrons = Particles(npmax, charge, mass)

    # Assign particles to subdomains
    electrons.initialize(x, y, vx, vy, grid)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    # assert comm.allreduce(electrons.np, op=MPI.SUM) == np

    # Set the electric field to zero
    E = Field(grid, comm, dtype=Float2)
    E.fill((0.0, 0.0))

    # Initialize sources
    sources = Sources(grid, comm, dtype=Float)

    # Initialize Poisson solver
    poisson = Poisson(grid, ax, ay, np)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit_ppic2(electrons)
    sources.rho.add_guards_ppic2()
    sources.rho += n0*npc

    # Solve Gauss' law
    poisson(sources.rho, E, destroy_input=False)
    # Set boundary condition
    E.copy_guards_ppic2()

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return numpy.concatenate(comm.allgather(arr))

    global_E = concatenate(E.trim())

    #
    E_pot = numpy.ones(nt)*1e-16
    time = numpy.arange(0, dt*nt, dt)

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings
        global_rho = concatenate(sources.rho.trim())

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest',
                   aspect='auto')
            plt.figure(1)
            fig, (ax1, ax2, ax3) = plt.subplots(num=1, nrows=3)
            im1 = ax1.imshow(global_rho)
            im2 = ax2.imshow(global_E['x'])
            im3 = ax3.imshow(global_E['y'])
            ax1.set_title(r'$\rho$')
            ax2.set_title(r'$E_x$')
            ax3.set_title(r'$E_y$')

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

        # Boundary calls
        sources.rho.add_guards_ppic2()
        sources.rho += n0

        # Solve Gauss' law
        poisson(sources.rho, E, destroy_input=False)

        # Set boundary condition
        E.copy_guards_ppic2()

        # sum(|E|) on each processor
        E_pot_id = (numpy.sqrt(E['x']**2 + E['y']**2)).sum()

        # Add contribution from each processor
        E_pot[it] = comm.allreduce(E_pot_id, op=MPI.SUM)

        # Make figures
        if plot:
            if (it % 1 == 0):
                global_rho = concatenate(sources.rho.trim())
                global_E = concatenate(E.trim())
                if comm.rank == 0:
                    im1.set_data(global_rho)
                    im2.set_data(global_E['x'])
                    im3.set_data(global_E['y'])
                    im1.autoscale()
                    im2.autoscale()
                    im3.autoscale()
                    plt.draw()
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)

    # Test if growth rate is correct
    if comm.rank == 0:
        from scipy.optimize import curve_fit

        # Exponential growth function
        def func(x, a, b):
            return a*numpy.exp(b*x)

        def lin_func(x, a, b):
            return a + b*x

        # Fit exponential to the evolution of sqrt(mean(B_x**2))
        # Disregard first half of data
        first = int(0.34*nt)
        last = int(0.6*nt)
        popt, pcov = curve_fit(lin_func, time[first:last],
                               numpy.log(E_pot[first:last]))

        # Theoretical gamma (TODO: Solve dispersion relation here)
        gamma_t = 0.352982

        # Gamma from the fit
        gamma_f = popt[1]

        # Relative error
        err = abs((gamma_f-gamma_t))/gamma_t

        # Tolerance
        tol = 4e-2

        # Did it work?
        assert (err < tol)

        if plot or fitplot:
            import matplotlib.pyplot as plt
            # Create figure
            plt.figure(3)
            plt.clf()
            plt.semilogy(time, E_pot, 'b')
            plt.semilogy(time[first:last], func(time[first:last],
                         numpy.exp(popt[0]), popt[1]), 'r--',
                         label=r"Fit: $\gamma = %.5f$" % popt[1])
            plt.semilogy(time, func(time, 1, gamma_t), 'k-',
                         label=r"Theory: $\gamma = %.5f$" % gamma_t)
            plt.xlabel("time")
            plt.legend(loc=2)
            plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--fitplot', '-fp', action='store_true')
    args = parser.parse_args()

    test_twostream(plot=args.plot, fitplot=args.fitplot)