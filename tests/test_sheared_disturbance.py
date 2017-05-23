from skeletor import Float, Float3, Particles, Sources
from skeletor import Field
from skeletor.manifolds.second_order import ShearingManifold
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_sheared_disturbance(plot=False):

    # Quiet start
    quiet = True

    # Time step
    dt = 0.5e-2

    # Initial time of particle positions
    t = -np.pi/2

    # Simulation time
    tend = np.pi/2

    # Number of time steps
    nt = int((tend-t)/dt)

    # Particle charge and mass
    charge = 1
    mass = 1

    # Keplerian frequency
    Omega = 1

    # Shear parameter
    S = -3/2

    # Epicyclic frequency
    kappa = np.sqrt(2*Omega*(2*Omega+S))

    # Amplitude of perturbation
    ampl = 0.03125

    # Number of grid points in x- and y-direction
    nx, ny = 64, 64

    # Box size
    Lx, Ly = 1.0, 1.0

    # Coordinate origin
    x0, y0 = -Lx/2, -Ly/2

    # Average number of particles per cell
    npc = 16

    # Total number of particles in simulation
    N = npc*nx*ny

    def x_an(ap, bp, t):
        phi = kx*ap
        x = 2*Omega/kappa*ampl*(np.sin(kappa*t + phi) - np.sin(phi)) + \
            ap - S*t*(bp - ampl*np.cos(phi))
        return x

    def y_an(ap, bp, t):
        phi = kx*ap
        y = ampl*(np.cos(kappa*t + phi) - np.cos(phi)) + bp
        return y

    def vx_an(ap, bp, t):
        phi = kx*ap
        vx = 2*Omega*ampl*np.cos(kappa*t + phi) \
            - S*(bp - ampl*np.cos(phi))
        return vx

    def vy_an(ap, bp, t):
        phi = kx*ap
        vy = -ampl*kappa*np.sin(kappa*t + phi)
        return vy

    def euler(ap, bp, t):
        return x_an(ap, bp, t) + S*t*y_an(ap, bp, t)

    def euler_prime(a, t):
        phi = kx*a
        dxda = 2*Omega/kappa*ampl*kx*(
             np.cos(kappa*t + phi) - np.cos(phi)) \
            + 1 - S*t*ampl*kx*np.sin(phi)
        dyda = -ampl*kx*(np.sin(kappa*t + phi) - np.sin(phi))

        return dxda + S*t*dyda

    def mean(f, axis=None):
        """Compute mean of an array across processors."""
        result = np.mean(f, axis=axis)
        if axis is None or axis == 0:
            # If the mean is to be taken over *all* axes or just the y-axis,
            # then we need to communicate
            result = comm.allreduce(result, op=MPI.SUM)/comm.size
        return result

    def rms(f):
        """Compute root-mean-square of an array across processors."""
        return np.sqrt(mean(f**2))

    def lagrange(xp, t, tol=1.48e-8, maxiter=50):
        """
        Given the Eulerian coordinate x' = x + Sty and time t, this function
        solves the definition x' = euler(a, t) for the Lagrangian coordinate a
        via the Newton-Raphson method.
        """
        # Use Eulerian coordinate as initial guess
        a = xp.copy()
        for it in range(maxiter):
            f = euler(a, 0, t) - xp
            df = euler_prime(a, t)
            b = a - f/df
            # This is not the safest criterion, but seems good enough
            err = rms(a - b)
            if err < tol:
                return b
            a = b.copy()
        msg = "maxiter={} exceeded without reaching tol={}. Solution w. rms={}"
        raise RuntimeError(msg.format(maxiter, tol, err))

    def rho_an(a, t):
        return 1/euler_prime(a, t)

    def find_a(x, y, t):
        """
        Calculate the Lagrangian coordinate, a, as a function of Eulerian grid
        position, (x, y), and time, t. Accepts 2D arrays for x and y.
        """
        xp = x + S*y*t
        a = lagrange(xp, t)
        return a

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = ShearingManifold(nx, ny, comm,
                                S=S, Omega=Omega, x0=x0, y0=y0, Lx=Lx, Ly=Ly)

    # x- and y-grid
    xx, yy = np.meshgrid(manifold.x, manifold.y)

    # Wave numbers
    kx = 2*np.pi/manifold.Lx

    # Maximum number of ions in each partition
    # Set to big number to make sure particles can move between grids
    Nmax = int(2*N/comm.size)

    # Create particle array
    ions = Particles(manifold, Nmax, time=dt/2, charge=charge, mass=mass)

    if quiet:
        # Uniform distribution of particle positions (quiet start)
        sqrt_npc = int(np.sqrt(npc))
        assert sqrt_npc**2 == npc, "'npc' must be a square of an integer."
        a, b = [ab.flatten().astype(Float) for ab in np.meshgrid(
            x0 + (np.arange(nx*sqrt_npc) + 0.5)*manifold.dx/sqrt_npc,
            y0 + (np.arange(ny*sqrt_npc) + 0.5)*manifold.dy/sqrt_npc)]
    else:
        a = x0 + Lx*np.random.uniform(size=N).astype(Float)
        b = y0 + Ly*np.random.uniform(size=N).astype(Float)

    # Assign particles to subdomains
    ions.initialize(a, b, a*0, b*0, b*0)

    # Set initial condition for particles
    # Position and velocities for this subdomain only
    x_sub = x0 + ions['x'][:ions.N]*manifold.dx
    y_sub = y0 + ions['y'][:ions.N]*manifold.dy
    # Particle positions at time=t
    ions['x'][:ions.N] = (x_an(x_sub, y_sub, t) - x0)/manifold.dx
    ions['y'][:ions.N] = (y_an(x_sub, y_sub, t) - y0)/manifold.dy
    # Particle velocities at time = t-dt/2
    ions['vx'][:ions.N] = vx_an(x_sub, y_sub, t-dt/2)
    ions['vy'][:ions.N] = vy_an(x_sub, y_sub, t-dt/2)
    ions.time = t
    ions.shear_periodic_y()
    ions.periodic_x()

    # Make sure particles actually reside in the local subdomain
    assert all(ions["y"][:ions.N] >= manifold.edges[0])
    assert all(ions["y"][:ions.N] < manifold.edges[1])

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(ions.N, op=MPI.SUM) == N

    # Initialize sources
    sources = Sources(manifold)
    rho_periodic = Field(manifold, time=0, dtype=Float)
    Jx_periodic = Field(manifold, time=0, dtype=Float)
    Jy_periodic = Field(manifold, time=0, dtype=Float)

    # Deposit sources
    sources.deposit(ions)
    assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
    sources.add_guards()
    assert np.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), N*charge/npc)
    sources.copy_guards()

    # Copy density into a shear field
    rho_periodic.active = sources.rho.trim()
    Jx_periodic.active = sources.Jx.trim()
    Jy_periodic.active = sources.Jy.trim()

    ag = manifold.x

    # Electric field
    E = Field(manifold, dtype=Float3)
    E.fill((0.0, 0.0, 0.0))
    E.copy_guards()

    B = Field(manifold, dtype=Float3)
    B.fill((0.0, 0.0, 0.0))
    B.copy_guards()

    def concatenate(arr):
        """Concatenate local arrays to obtain global arrays
        The result is available on all processors."""
        return np.concatenate(comm.allgather(arr))

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings

        global_rho = concatenate(sources.rho.trim())
        global_rho_periodic = concatenate(rho_periodic.trim())
        global_Jx = concatenate(sources.Jx.trim())
        global_Jx_periodic = concatenate(Jx_periodic.trim())
        global_Jy = concatenate(sources.Jy.trim())
        global_Jy_periodic = concatenate(Jy_periodic.trim())

        if comm.rank == 0:
            plt.rc('image', origin='upper', interpolation='nearest',
                   cmap='coolwarm')
            plt.figure(1)
            plt.clf()
            fig, axes = plt.subplots(num=1, ncols=2, nrows=3)
            im1a = axes[0, 0].imshow(global_rho)
            im2a = axes[0, 1].imshow(global_rho_periodic)
            im1b = axes[1, 0].imshow(global_Jx/global_rho)
            im2b = axes[1, 1].imshow(global_Jx_periodic/global_rho_periodic)
            im1c = axes[2, 0].imshow(global_Jy/global_rho)
            im2c = axes[2, 1].imshow(global_Jy_periodic/global_rho_periodic)
            plt.figure(2)
            plt.clf()
            fig2, (ax1, ax2, ax3) = plt.subplots(num=2, nrows=3)
            im4 = ax1.plot(manifold.x, (global_rho_periodic.mean(axis=0)),
                           'b',
                           manifold.x, (global_rho_periodic.mean(axis=0)),
                           'r--')
            vx = global_Jx_periodic/global_rho_periodic
            im5 = ax2.plot(manifold.x, vx.mean(axis=0), 'b',
                           manifold.x, vx.mean(axis=0), 'r--')
            vy = global_Jy_periodic/global_rho_periodic
            im6 = ax3.plot(manifold.x, vy.mean(axis=0), 'b',
                           manifold.x, vy.mean(axis=0), 'r--')
            ax1.set_ylim(0.5, 1.8)
            ax2.set_ylim(-1*ampl, 1*ampl)
            ax3.set_ylim(-2*ampl, 2*ampl)
            for ax in (ax1, ax2, ax3):
                ax.set_xlim(x0, x0 + manifold.Lx)

    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################

    for it in range(nt):
        # Deposit sources
        sources.deposit(ions)
        sources.time = t
        sources.add_guards()
        sources.copy_guards()

        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        ions.push_modified(E, B, dt)

        # Update time
        t += dt

        # Copy density into a shear field
        rho_periodic.active = sources.rho.trim()
        Jx_periodic.active = sources.Jx.trim()
        Jy_periodic.active = sources.Jy.trim()

        # Translate the density to be periodic in y
        rho_periodic.translate(-t)
        rho_periodic.copy_guards()

        Jx_periodic.translate(-t)
        Jx_periodic.copy_guards()

        Jy_periodic.translate(-t)
        Jy_periodic.copy_guards()

        # Make figures
        if (it % 60 == 0):
            # Calculate rms of numerical solution wrt to analytical solution
            # a_2d = find_a(xx, yy, t)
            # err = rms(sources.rho.trim() - rho_an(a_2d, t))
            # Check if test is passed
            # assert err < 1e-2, err
            if plot:
                global_rho = concatenate(sources.rho.trim())
                global_rho_periodic = concatenate(rho_periodic.trim())
                global_Jx = concatenate(sources.Jx.trim())
                global_Jx_periodic = concatenate(Jx_periodic.trim())
                global_Jy = concatenate(sources.Jy.trim())
                global_Jy_periodic = concatenate(Jy_periodic.trim())

                if comm.rank == 0:
                    im1a.set_data(global_rho)
                    im2a.set_data(global_rho_periodic)
                    im1b.set_data(global_Jx/global_rho)
                    im2b.set_data(global_Jx_periodic/global_rho_periodic)
                    im1c.set_data(global_Jy/global_rho)
                    im2c.set_data(global_Jy_periodic/global_rho_periodic)
                    im1a.autoscale()
                    im2a.autoscale()
                    im1b.autoscale()
                    im2b.autoscale()
                    im1c.autoscale()
                    im2c.autoscale()
                    im4[0].set_ydata(global_rho_periodic.mean(axis=0))
                    vx = global_Jx_periodic/global_rho_periodic
                    im5[0].set_ydata(vx.mean(axis=0))
                    vy = global_Jy_periodic/global_rho_periodic
                    im6[0].set_ydata(vy.mean(axis=0))
                    xp_par = euler(ag, 0, t)
                    xp_par = (xp_par - x0) % manifold.Lx + x0
                    ind = np.argsort(xp_par)
                    im4[1].set_data(xp_par[ind], rho_an(ag, t)[ind])
                    im5[1].set_data(xp_par[ind], vx_an(ag, 0, t)[ind]
                                    + S*y_an(ag, 0, t)[ind])
                    im6[1].set_data(xp_par[ind], vy_an(ag, 0, t)[ind])

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_sheared_disturbance(plot=args.plot)
