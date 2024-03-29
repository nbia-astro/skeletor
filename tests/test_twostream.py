from skeletor import Float, Float3, Field, Particles, Sources
from skeletor.manifolds.second_order import Manifold
from skeletor import Poisson
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_twostream(plot=False, fitplot=False):
    """Test of twostream instability."""

    quiet = True

    # Number of grid points in x- and y-direction
    nx, ny = 64, 4

    # Average number of particles per cell
    npc = 32

    # Number of time steps
    nt = 1000

    # Particle charge and mass
    charge = -1.0
    mass = 1.0

    # Timestep
    dt = 0.05

    # Total number of particles in simulation
    N = npc*nx*ny

    # Mean velocity of electrons in x-direction
    vdx, vdy = 1/10, 0.

    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 0., 0.

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm)

    # wavenumber
    kx = 2*np.pi/manifold.Lx

    if quiet:
        # Uniform distribution of particle positions (quiet start)
        sqrt_npc = int(np.sqrt(npc//2))
        assert (sqrt_npc)**2*2 == npc
        x1 = (np.arange(manifold.nx*sqrt_npc) + 0.5)*manifold.dx/sqrt_npc
        y1 = (np.arange(manifold.ny*sqrt_npc) + 0.5)*manifold.dy/sqrt_npc
        x, y = [xy.flatten() for xy in np.meshgrid(x1, y1)]
    else:
        x = nx*np.random.uniform(size=N).astype(Float)
        y = ny*np.random.uniform(size=N).astype(Float)

    vx = vdx*np.ones_like(x)
    vy = vdy*np.ones_like(y)

    # Have two particles at position
    x = np.concatenate([x, x])
    y = np.concatenate([y, y])

    # Small perturbation to get the instability going
    x += 1e-4*manifold.dx*np.cos(kx*x)

    # Make counterpropagating in x
    vx = np.concatenate([vx, -vx])
    vy = np.concatenate([vy, vy])

    # Add thermal component
    vx += vtx*np.random.normal(size=N).astype(Float)
    vy += vty*np.random.normal(size=N).astype(Float)
    vz = np.zeros_like(vx)

    # Maximum number of electrons in each partition
    Nmax = int(1.5*N/comm.size)

    # Create particle array
    electrons = Particles(manifold, Nmax, charge=charge, mass=mass)

    # Assign particles to subdomains
    electrons.initialize(x, y, vx, vy, vz)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    # assert comm.allreduce(electrons.N, op=MPI.SUM) == N

    # Set the electric field to zero
    E = Field(manifold, comm, dtype=Float3)
    E.fill((0.0, 0.0, 0.0))

    B = Field(manifold, dtype=Float3)
    B.fill((0.0, 0.0, 0.0))
    B.copy_guards()

    # Initialize sources
    sources = Sources(manifold)

    # Initialize Poisson solver
    poisson = Poisson(manifold)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(electrons)
    sources.add_guards()

    # Solve Gauss' law
    poisson(sources.rho, E)
    # Set boundary condition
    E.copy_guards()

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return np.concatenate(comm.allgather(arr))

    global_E = concatenate(E.trim())

    #
    E_pot = np.ones(nt)*1e-16
    time = np.arange(0, dt*nt, dt)

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
            im3 = ax3.plot(electrons['x'][:N], electrons['vx'][:N], 'o',
                           fillstyle='full', ms=1)
            ax1.set_title(r'$\rho$')
            ax2.set_title(r'$E_x$')
            ax3.set_ylim(-2*vdx, 2*vdx)
            ax3.set_xlim(0, nx)
            for ax in (ax1, ax2, ax3):
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
            ax3.set_ylabel(r'$v_x$')

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

        # Boundary calls
        sources.add_guards()

        # Solve Gauss' law
        poisson(sources.rho, E)

        # Set boundary condition
        E.copy_guards()

        # sum(|E|) on each processor
        E_pot_id = (np.sqrt(E['x']**2 + E['y']**2)).sum()

        # Add contribution from each processor
        E_pot[it] = comm.allreduce(E_pot_id, op=MPI.SUM)

        # Make figures
        if plot:
            if (it % 10 == 0):
                global_rho = concatenate(sources.rho.trim())
                global_E = concatenate(E.trim())
                if comm.rank == 0:
                    im1.set_data(global_rho)
                    im2.set_data(global_E['x'])
                    im3[0].set_data(electrons['x'][:N], electrons['vx'][:N])
                    im1.autoscale()
                    im2.autoscale()
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
            return a*np.exp(b*x)

        def lin_func(x, a, b):
            return a + b*x

        # Fit exponential to the evolution of sqrt(mean(B_x**2))
        # Disregard first half of data
        first = int(0.34*nt)
        last = int(0.6*nt)
        popt, pcov = curve_fit(lin_func, time[first:last],
                               np.log(E_pot[first:last]))

        # Gamma from the fit
        gamma_f = popt[1]

        # Plasma frequency (squared) of each beam
        # Note the factor of 1/2, which is due to the fact that electrons.n0 is
        # the *total* number density of the two beams combined.
        # Note also that we're working in units where the vacuum permittivity
        # ε0 has been scaled out.
        wp2 = charge*charge*0.5*electrons.n0/mass
        # Doppler frequency (squared)
        kv2 = kx*kx*vdx*vdx
        # The dispersion relation of the two stream instability with two
        # counter-propagating beams with velocities v and -v and having equal
        # number densities is
        #   (ω² - k²v²)² - 2(k²v² + ω²) ωp² = 0,
        # where
        #   ωp² = e²n/m
        # is the plasma frequency of each beam. The above dispersion relation
        # is a quadratic equation in ω². The unstable root is
        w2 = kv2 + wp2 - np.sqrt(wp2*(4*kv2 + wp2))
        # The corresponding growth rate is
        gamma_t = np.sqrt(-w2)

        # Relative error
        err = abs((gamma_f-gamma_t))/gamma_t

        # Tolerance
        tol = 4e-3

        # Did it work?
        assert (err < tol)

        if plot or fitplot:
            import matplotlib.pyplot as plt
            # Create figure
            plt.figure(3)
            plt.clf()
            plt.semilogy(time, E_pot, 'b')
            plt.semilogy(time[first:last], func(time[first:last],
                         np.exp(popt[0]), popt[1]), 'r--',
                         label=r"Fit: $\gamma = %.5f$" % popt[1])
            plt.semilogy(time, func(time, 1, gamma_t), 'k-',
                         label=r"Theory: $\gamma = %.5f$" % gamma_t)
            plt.xlabel("time")
            plt.ylabel(r"$E^2$")
            plt.legend(loc=2)
            plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--fitplot', '-fp', action='store_true')
    args = parser.parse_args()

    test_twostream(plot=args.plot, fitplot=args.fitplot)
