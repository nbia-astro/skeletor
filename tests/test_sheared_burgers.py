"""
This script tests the density field for a sheared disturbance against the
analytical solution. This disturbance correponds to the standard 1D Burgers'
equation in the primed coordinate, x' = x + Sty. See also test_burgers.py.
"""

from skeletor import Float, Float3, Particles, Sources, ShearField
from skeletor.manifolds.second_order import ShearingManifold
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_sheared_burgers(plot=False):

    # Time step
    dt = 0.5e-2

    # Initial time of particle positions
    t = -numpy.pi/3

    # Simulation time
    tend = numpy.pi/3

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
    ampl = 2.

    # Number of grid points in x- and y-direction
    nx, ny = 32, 32

    # Average number of particles per cell
    npc = 16

    # Wave numbers
    kx = 2*numpy.pi/nx

    # Total number of particles in simulation
    np = npc*nx*ny

    # Uniform distribution of particle positions (quiet start)
    sqrt_npc = int(numpy.sqrt(npc))
    assert sqrt_npc**2 == npc
    dx = dy = 1/sqrt_npc
    a, b = numpy.meshgrid(
            numpy.arange(dx/2, nx+dx/2, dx),
            numpy.arange(dy/2, ny+dy/2, dy))
    a = a.flatten()
    b = b.flatten()

    def mean(f, axis=None):
        """Compute mean of an array across processors."""
        result = numpy.mean(f, axis=axis)
        if axis is None or axis == 0:
            # If the mean is to be taken over *all* axes or just the y-axis,
            # then we need to communicate
            result = comm.allreduce(result, op=MPI.SUM)/comm.size
        return result

    def rms(f):
        """Compute root-mean-square of an array across processors."""
        return numpy.sqrt(mean(f**2))

    def velocity(a):
        """Particle velocity in Lagrangian coordinates."""
        return ampl*numpy.sin(kx*a)

    def velocity_prime(a):
        """Derivative of particle velocity in Lagrangian coordinates:
        ∂v(a,τ)/∂a"""
        return ampl*kx*numpy.cos(kx*a)

    def euler(a, τ):
        """
        This function converts from Lagrangian to Eulerian sheared coordinate
        x' by solving ∂x'(a, τ)/∂τ = U(a) for x'(a, τ) subject to the initial
        condition x'(a, 0) = a.
        """
        return a + velocity(a)*τ

    def euler_prime(a, τ):
        """
        The derivative ∂x'/∂a of the conversion function defined above, which
        is related to the mass density in Lagrangian coordinates through
        rho(a, τ)/rho_0(a) = (∂x'/∂a)⁻¹, where rho_0(a) = rho(a, 0) is the
        initial mass density.
        """
        return 1 + velocity_prime(a)*τ

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
        xp %= nx
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
    manifold = ShearingManifold(nx, ny, comm, S=S, Omega=Omega)

    # x- and y-grid
    xx, yy = numpy.meshgrid(manifold.x, manifold.y)

    # Maximum number of ions in each partition
    # Set to big number to make sure particles can move between grids
    npmax = int(5*np/comm.size)

    # Create particle array
    ions = Particles(manifold, npmax, time=t, charge=charge, mass=mass)

    # Assign particles to subdomains (zero velocity and uniform distribution)
    ions.initialize(a, b, a*0, b*0, b*0)

    # Position and velocities for this subdomain only
    x_sub = numpy.copy(ions['x'][:ions.np])
    y_sub = numpy.copy(ions['y'][:ions.np])

    # Set initial condition
    ions['vx'][:ions.np] = vx_an(x_sub, y_sub, t)
    ions['x'][:ions.np] = x_an(x_sub, y_sub, t)

    # Set boundary condition on particles
    ions.time = t
    ions.shear_periodic_y()
    ions.periodic_x()

    # Make sure particles actually reside in the local subdomain
    assert all(ions["y"][:ions.np] >= manifold.edges[0])
    assert all(ions["y"][:ions.np] < manifold.edges[1])

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Initialize sources
    sources = Sources(manifold, npc)
    sources.rho = ShearField(manifold, time=0, dtype=Float)
    rho_periodic = ShearField(manifold, time=0, dtype=Float)
    J_periodic = ShearField(manifold, time=0, dtype=Float3)

    # Deposit sources
    sources.deposit(ions)
    assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
    sources.rho.add_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)
    sources.rho.copy_guards()

    # Copy density into a shear field
    rho_periodic.active = sources.rho.trim()

    # Set the electric field to zero
    E = ShearField(manifold, dtype=Float3)
    E.fill((0.0, 0.0, 0.0))
    E.copy_guards()

    B = ShearField(manifold, dtype=Float3)
    B.fill((0.0, 0.0, 0.0))
    B.copy_guards()

    def concatenate(arr):
        """Concatenate local arrays to obtain global arrays
        The result is available on all processors."""
        return numpy.concatenate(comm.allgather(arr))

    global_rho = concatenate(sources.rho.trim())
    global_rho_periodic = concatenate(rho_periodic.trim())
    global_J = concatenate(sources.J.trim())
    global_J_periodic = concatenate(J_periodic.trim())

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
            im1b = axes[1, 0].imshow(global_J['x']/global_rho)
            im2b = axes[1, 1].imshow(global_J_periodic['x']
                                     / global_rho_periodic)

            plt.figure(2)
            plt.clf()
            fig2, (ax1, ax2) = plt.subplots(num=2, nrows=2)
            ax1.set_ylim(0, 2)
            ax2.set_ylim(-1*ampl, 1*ampl)
            for ax in (ax1, ax2):
                ax.set_xlim(0, nx)
            ax1.set_xlabel(r"$x'$")
            ax1.set_title(r'$\rho/\rho_0$')
            # Create slider widget for changing time
            xp = euler(manifold.x, 0)
            xp = numpy.sort(xp)
            im4 = ax1.plot(manifold.x, (global_rho_periodic.mean(axis=0)),
                           'b',
                           manifold.x, (global_rho_periodic.mean(axis=0)),
                           'r--')
            im5 = ax2.plot(manifold.x, (global_J_periodic['x']
                           / global_rho_periodic).mean(axis=0), 'b',
                           manifold.x, (global_J_periodic['x']
                           / global_rho_periodic).mean(axis=0), 'r--')

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
        sources.rho.time = t
        sources.J.time = t

        sources.rho.add_guards()
        sources.J.add_guards()

        sources.rho.copy_guards()
        sources.J.copy_guards()

        # Copy density into a shear field
        rho_periodic.active = sources.rho.trim()
        J_periodic.active = sources.J.trim()

        # Translate the density to be periodic in y
        rho_periodic.translate(-t)
        rho_periodic.copy_guards()

        J_periodic.translate(-t)
        J_periodic.copy_guards()

        # Calculate rms of numerical solution wrt to the analytical solution
        err = rms(sources.rho.trim() - rho2d_an(xx, yy, t))

        global_rho = concatenate(sources.rho.trim())
        global_rho_periodic = concatenate(rho_periodic.trim())
        global_J = concatenate(sources.J.trim())
        global_J_periodic = concatenate(J_periodic.trim())

        if comm.rank == 0:
            # Make sure that test has passed
            assert err < 1e-2, err
            if (it % 100 == 0):
                if plot:
                    # Update 2D images
                    im1a.set_data(global_rho)
                    im2a.set_data(global_rho_periodic)
                    im1b.set_data(global_J['x']/global_rho)
                    im2b.set_data(global_J_periodic['x']/global_rho_periodic)
                    for im in (im1a, im2a, im1b, im2b):
                        im.autoscale()
                    # Update 1D solutions (numerical and analytical)
                    im4[0].set_ydata(global_rho_periodic.mean(axis=0))
                    im5[0].set_ydata((global_J_periodic['x']
                                     / global_rho_periodic).mean(axis=0))
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
