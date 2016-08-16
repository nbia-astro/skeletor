from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
from skeletor import Ohm
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def landau_ions(plot=False, fitplot=False):

    # Quiet start
    quiet = True
    # Number of grid points in x- and y-direction
    nx, ny = 32, 1
    # Average number of particles per cell
    npc = 2**16
    # Particle charge and mass
    charge = 1
    mass = 1.0
    # Electron temperature
    Te = 1.0
    # Ion temperature
    Ti = 1/5
    # Dimensionless amplitude of perturbation
    A = 0.01
    # Wavenumbers
    ikx = 1
    iky = 0
    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = numpy.sqrt(Ti/mass), 0.0
    # CFL number
    cfl = 0.5
    # Number of periods to run for
    nperiods = 0.5

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
    nt = 190#int(tend/dt)

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

    # Set the electric field to zero
    E = Field(grid, comm, dtype=Float2)
    E.fill((0.0, 0.0))

    # Initialize sources
    sources = Sources(grid, comm, dtype=Float)

    # Initialize Ohm's law solver
    ohm = Ohm(grid, npc, temperature=Te, charge=charge)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(ions)
    assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    sources.rho.add_guards_ppic2()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge)

    # Calculate electric field (Solve Ohm's law)
    ohm(sources.rho, E, destroy_input=False)
    # Set boundary condition
    E.copy_guards_ppic2()

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return numpy.concatenate(comm.allgather(arr))

    global_E = concatenate(E.trim())

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings
        global_rho = concatenate(sources.rho.trim())

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest',aspect='auto')
            plt.figure(1)
            fig, (ax1, ax2, ax3) = plt.subplots(num=1, nrows=3)
            im1 = ax1.imshow(global_rho)
            im2 = ax2.imshow(global_E['x'])
            im3 = ax3.imshow(global_E['y'])
            ax1.set_title(r'$\rho$')
            ax2.set_title(r'$E_x$')
            ax3.set_title(r'$E_y$')

    #
    drho    = numpy.ones(nt)*1e-16
    drho_k  = numpy.ones(nt)*1e-16
    time    = numpy.arange(0,dt*nt, dt)

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

        # Boundary calls
        sources.rho.add_guards_ppic2()

        # Calculate forces (Solve Ohm's law)
        ohm(sources.rho, E, destroy_input=False)
        # Set boundary condition
        E.copy_guards_ppic2()

        # sum(|drho|) on each processor
        drho_i = (numpy.sqrt((sources.rho.trim()-rho_an(xg, yg, 0))**2)).sum()

        drho_k_i = (abs (numpy.fft.rfft (sources.rho.trim (), axis=1)[0,ikx])/nx).mean()

        # Add contribution from each processor
        drho[it]   = comm.allreduce(drho_i, op=MPI.SUM)
        drho_k[it] = comm.allreduce(drho_k_i, op=MPI.SUM)

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
        from scipy.signal import argrelextrema

        # Find local maxima
        index = argrelextrema(drho_k, numpy.greater)
        tmax = time[index]
        ymax = drho_k[index]

        # Exponential growth function
        def func(x,a,b):
            return a*numpy.exp(b*x)

        def lin_func(x,a,b):
            return a + b*x

        # Fit the local maxima
        popt, pcov = curve_fit(lin_func, tmax, numpy.log(ymax))

        # Theoretical gamma (TODO: Solve dispersion relation here)
        gamma_t = -0.0203927225606

        # Gamma from the fit
        gamma_f = popt[1]

        # Relative error
        err = abs((gamma_f-gamma_t))/gamma_t

        # Tolerance
        tol = 2e-2

        # Did it work?
        print (err)
        # assert (err < tol)

        if plot or fitplot:
            import matplotlib.pyplot as plt
            # Create figure
            plt.figure (3)
            plt.clf ()
            plt.semilogy(time, drho_k, 'b')
            plt.semilogy(tmax, ymax,   'r*')
            plt.semilogy(tmax,func(tmax,numpy.exp(popt[0]),popt[1]),
              'r--',label=r"Fit: $\gamma_f = %.5f$" %popt[1])
            plt.semilogy(time,func(time,ymax[0]*numpy.exp(-tmax[0]*gamma_t)
                ,gamma_t),'k-',label=r"Theory: $\gamma_t = %.5f$" %gamma_t)

            plt.title (r'$|\hat{\rho}(k=2\pi/L)|$')
            plt.legend(loc=3)
            plt.xlabel("time")
            # plt.savefig("landau-damping.pdf")
            plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--fitplot', '-fp', action='store_true')
    args = parser.parse_args()

    landau_ions(plot=args.plot, fitplot=args.fitplot)
