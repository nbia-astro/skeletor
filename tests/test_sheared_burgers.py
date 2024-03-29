"""
This script tests the density field for a sheared disturbance against the
analytical solution. This disturbance correponds to the standard 1D Burgers'
equation in the primed coordinate, x' = x + Sty. See also test_burgers.py.
"""

from skeletor import Float, Float3, Particles, Sources, Field
from skeletor.manifolds.second_order import ShearingManifold
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_sheared_burgers(plot=False):

    # Order of particle interpolation
    order = 1

    # Required number of guard layers on each side
    ghost = order//2 + 1

    # Time step
    dt = 0.5e-2

    # Initial time of particle positions
    t = -np.pi/3

    # Simulation time
    tend = np.pi/3

    # Number of time steps
    nt = int((tend-t)/dt)

    # Particle charge and mass
    charge = 1
    mass = 1

    # Keplerian frequency
    Omega = 0

    # Shear parameter
    S = -3/2

    # Amplitude of perturbation
    ampl = 0.0625

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

    def velocity(a):
        """Particle velocity in Lagrangian coordinates."""
        return ampl*np.sin(kx*a)

    def velocity_prime(a):
        """Derivative of particle velocity in Lagrangian coordinates:
        ∂v(a,τ)/∂a"""
        return ampl*kx*np.cos(kx*a)

    def euler(a, tau):
        """
        This function converts from Lagrangian to Eulerian sheared coordinate
        x' by solving ∂x'(a, τ)/∂τ = U(a) for x'(a, τ) subject to the initial
        condition x'(a, 0) = a.
        """
        return a + velocity(a)*tau

    def euler_prime(a, tau):
        """
        The derivative ∂x'/∂a of the conversion function defined above, which
        is related to the mass density in Lagrangian coordinates through
        rho(a, τ)/rho_0(a) = (∂x'/∂a)⁻¹, where rho_0(a) = rho(a, 0) is the
        initial mass density.
        """
        return 1 + velocity_prime(a)*tau

    def lagrange(xp, t, tol=1.48e-8, maxiter=50):
        """
        Given the Eulerian coordinate x' = x + Sty and time t, this function
        solves the definition x' = euler(a, t) for the Lagrangian coordinate a
        via the Newton-Raphson method.
        """
        # Use Eulerian coordinate as initial guess
        a = xp.copy()
        for it in range(maxiter):
            f = euler(a, t) - xp
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
        """
        Calculates the analytical density as a function of the lagragian
        coordiate, a.
        """
        return 1/euler_prime(a, t)

    def rho2d_an(x, y, t):
        """
        Calculate the analytical density as a function of Eulerian grid
        position, (x, y), and time, t. Accepts 2D arrays for x and y.
        """
        xp = x + S*y*t
        xp = (xp - x0) % manifold.Lx + x0
        a = lagrange(xp, t)
        return rho_an(a, t)

    def vx_an(a, b, t):
        """Particle velocity along x is perturbation plus shear"""
        vx = velocity(a) - b*S
        return vx

    def x_an(a, b, t):
        """Particle x-position as a function of time"""
        return a + vx_an(a, b, t)*t

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = ShearingManifold(nx, ny, comm, lbx=ghost, lby=ghost,
                                S=S, Omega=Omega, x0=x0, y0=y0, Lx=Lx, Ly=Ly)

    # x- and y-grid
    xx, yy = np.meshgrid(manifold.x, manifold.y)

    # Wave numbers
    kx = 2*np.pi/manifold.Lx

    # Maximum number of ions in each partition
    # Set to big number to make sure particles can move between grids
    Nmax = int(5*N/comm.size)

    # Create particle array
    ions = Particles(manifold, Nmax, time=t, charge=charge, mass=mass,
                     order=order)

    # Lagrangian particle coordinates (quiet start)
    sqrt_npc = int(np.sqrt(npc))
    assert sqrt_npc**2 == npc, "'npc' must be a square of an integer."
    a, b = [ab.flatten().astype(Float) for ab in np.meshgrid(
        x0 + (np.arange(nx*sqrt_npc) + 0.5)*manifold.dx/sqrt_npc,
        y0 + (np.arange(ny*sqrt_npc) + 0.5)*manifold.dy/sqrt_npc)]

    # Eulerian particle coordinates and veloctities
    x = x_an(a, b, t)
    y = b
    vx = vx_an(a, b, t)
    vy = np.zeros_like(vx)
    vz = vy

    # Assign particles to subdomains (zero velocity and uniform distribution)
    ions.initialize(x, y, vx, vy, vz)

    # Set boundary condition on particles
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

    # Set the electric field to zero
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

    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(rho_periodic.trim())
    global_Jx = concatenate(sources.Jx.trim())
    global_Jx_periodic = concatenate(Jx_periodic.trim())

    if comm.rank == 0:

        if plot:
            import matplotlib.pyplot as plt
            from matplotlib.cbook import mplDeprecation
            import warnings

            plt.rc('image', origin='upper', interpolation='nearest',
                   cmap='coolwarm')
            plt.figure(1)
            plt.clf()
            fig, axes = plt.subplots(num=1, ncols=2, nrows=2)
            im1a = axes[0, 0].imshow(global_rho)
            im2a = axes[0, 1].imshow(global_rho_periodic)
            im1b = axes[1, 0].imshow(global_Jx/global_rho)
            im2b = axes[1, 1].imshow(global_Jx_periodic/global_rho_periodic)

            plt.figure(2)
            plt.clf()
            fig2, (ax1, ax2) = plt.subplots(num=2, nrows=2)
            ax1.set_ylim(0, 2)
            ax2.set_ylim(-1*ampl, 1*ampl)
            for ax in (ax1, ax2):
                ax.set_xlim(x0, x0 + Lx)
            ax1.set_xlabel(r"$x'$")
            ax1.set_title(r'$\rho/\rho_0$')
            # Create slider widget for changing time
            xp = euler(manifold.x, 0)
            xp = np.sort(xp)
            im4 = ax1.plot(manifold.x, (global_rho_periodic.mean(axis=0)),
                           'b',
                           manifold.x, (global_rho_periodic.mean(axis=0)),
                           'r--')
            vx = global_Jx_periodic/global_rho_periodic
            im5 = ax2.plot(manifold.x, vx.mean(axis=0), 'b',
                           manifold.x, vx.mean(axis=0), 'r--')

    for it in range(nt):
        # Updates the particle position and velocity, deposits the charge and
        # current, and produces two figures. Figure 1 shows rho and Jx, which
        # are shearing periodic, in the first column. In the second column it
        # shows the same fields but translated by a distance -S*t*y along x.
        # This makes the fields periodic.

        # The second figures then shows the analytical solutions for rho and
        # Jx as a function of the primed coordinate, x', which are compared
        # with the numerical solution found by averaging the periodic fields
        # shown in Figure 1 along y.

        # Push particles on each processor. This call also sends and receives
        # particles to and from other processors/subdomains.
        ions.push_modified(E, B, dt)

        # Update the time
        t += dt

        # Deposit sources
        sources.deposit(ions)
        sources.time = t

        sources.add_guards()
        sources.copy_guards()

        # Copy density into a shear field
        rho_periodic.active = sources.rho.trim()
        Jx_periodic.active = sources.Jx.trim()

        # Translate the density to be periodic in y
        rho_periodic.translate(-t)
        rho_periodic.copy_guards()

        Jx_periodic.translate(-t)
        Jx_periodic.copy_guards()

        # Calculate rms of numerical solution wrt to the analytical solution
        err = rms(sources.rho.trim() - rho2d_an(xx, yy, t))

        global_rho = concatenate(sources.rho.trim())
        global_rho_periodic = concatenate(rho_periodic.trim())
        global_Jx = concatenate(sources.Jx.trim())
        global_Jx_periodic = concatenate(Jx_periodic.trim())

        if comm.rank == 0:
            # Make sure that test has passed
            assert err < 1e-2, err
            if (it % 50 == 0):
                if plot:
                    # Update 2D images
                    im1a.set_data(global_rho)
                    im2a.set_data(global_rho_periodic)
                    im1b.set_data(global_Jx/global_rho)
                    im2b.set_data(global_Jx_periodic/global_rho_periodic)
                    for im in (im1a, im2a, im1b, im2b):
                        im.autoscale()
                    # Update 1D solutions (numerical and analytical)
                    im4[0].set_ydata(global_rho_periodic.mean(axis=0))
                    vx = global_Jx_periodic/global_rho_periodic
                    im5[0].set_ydata(vx.mean(axis=0))
                    xp = euler(manifold.x, t)
                    im4[1].set_data(xp, rho_an(manifold.x, t))
                    im5[1].set_data(xp, velocity(manifold.x))

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_sheared_burgers(plot=args.plot)
