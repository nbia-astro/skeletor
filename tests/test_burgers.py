"""
This test assumes that no force whatsoever is acting on the ions. In 1D, the
equation of motion is thus simply d²x/dt²=0. This means that the code
effectively solves the inviscid Burgers equation because the first moment is
∂u/∂t + u ∂u/∂x = 0. While not needed to determine the evolution of the system,
the continuity equation is ∂ρ/∂t + ∂(ρu)/∂x = 0.

Initially, the density profile is uniform and the velocity profile is
sinusoidal. As the particles stream freely, they are scattered onto the grid
and the resulting mass (or number) density is compared with the exact solution.
The latter is known in Lagrangian coordinates. To enable a direct,
*quantitative* comparison with the solution the code produces, the Eulerian
grid coordinates are first converted into Lagrangian coordinates (a one-to-one
map as long as solution is single-valued).

The simulation runs for a hundred or so time steps. The root-mean-squared error
between the exact and numerical solutions from each step are added up, averaged
in the end, and verified to be reasonably small (currently < 1e-3).
"""

from skeletor import Float, Particles, Sources
from skeletor.manifolds.second_order import Manifold
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm


def test_burgers(plot=False):

    # Number of particles per cell
    npc = 64
    # Number of grid points
    nx, ny = 256, 4

    # Wave number and amplitude
    ikx = 1
    ampl = 0.1

    def mean(f, axis=None):
        "Compute mean of an array across processors."
        result = np.mean(f, axis=axis)
        if axis is None or axis == 0:
            # If the mean is to be taken over *all* axes or just the y-axis,
            # then we need to communicate
            result = comm.allreduce(result, op=MPI.SUM)/comm.size
        return result

    def rms(f):
        "Compute root-mean-square of an array across processors."
        return np.sqrt(mean(f**2))

    def velocity(a):
        "Particle velocity in Lagrangian coordinates."
        return ampl*np.sin(kx*a)

    def velocity_prime(a):
        "Derivative of particle velocity in Lagrangian coordinates: ∂v(a,τ)/∂a"
        return ampl*kx*np.cos(kx*a)

    def euler(a, τ):
        """
        This function converts from Lagrangian to Eulerian coordinate by
        solving ∂x(a, τ)/∂τ = u(a) for x(a, τ) subject to the initial condition
        x(a, 0) = a.
        """
        return (a + velocity(a)*τ) % manifold.Lx

    def euler_prime(a, τ):
        """
        The derivative ∂x/∂a of the conversion function defined above, which is
        related to the mass density in Lagrangian coordinates through
            rho(a, τ)/rho_0(a) = (∂x/∂a)⁻¹,
        where rho_0(a) = rho(a, 0) is the initial mass density.
        """
        return 1 + velocity_prime(a)*τ

    def lagrange(x, t, tol=1.48e-8, maxiter=50):
        """
        Given the Eulerian coordinate x and time t, this function solves the
        definition x = euler(a, t) for the Lagrangian coordinate a via the
        Newton-Raphson method.
        """
        # Use Eulerian coordinate as initial guess
        a = x.copy()
        for it in range(maxiter):
            f = euler(a, t) - x
            df = euler_prime(a, t)
            b = a - f/df
            # This is not the safest criterion, but seems good enough
            if rms(a - b) < tol:
                return b
            a = b.copy()

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm)

    # Initialize sources
    sources = Sources(manifold)

    # Maximum number of ions in each partition
    # Set to big number to make sure particles can move between grids
    npmax = int(1.25*npc*nx*ny/comm.size)

    # Create particle array
    ions = Particles(manifold, npmax, time=0.0, charge=1.0, mass=1.0)

    if plot:
        import matplotlib.pyplot as plt
        # Create figure
        plt.figure(1)
        plt.clf()
        fig, axis = plt.subplots(num=1)
        axis.set_ylim(0, 4)
        axis.set_xlabel(r'$x$')
        axis.set_title(r'$\rho/\rho_0$')
        fig.set_tight_layout(True)
        lines = axis.plot(manifold.x, np.ones_like(manifold.x), 'k',
                          manifold.x, np.ones_like(manifold.x), 'r--')

    # Initial time
    t = -1.171875  # = 300/nx
    # Time step
    dt = 0.015625  # = 4/nx
    # Number of time steps
    nt = 150

    # Lagrangian/labeling coordinates
    sqrt_npc = int(np.sqrt(npc))
    assert sqrt_npc**2 == npc, "'npc' must be a square of an integer."
    ax, ay = [ab.flatten().astype(Float) for ab in np.meshgrid(
        manifold.dx*(np.arange(nx*sqrt_npc) + 0.5)/sqrt_npc,
        manifold.dy*(np.arange(ny*sqrt_npc) + 0.5)/sqrt_npc
        )]

    # x-component of wave vector
    kx = 2*np.pi*ikx/manifold.Lx

    # Particle position (i.e. Eulerian coordinate) and velocity
    x = euler(ax, t)
    y = ay
    vx = velocity(ax)
    vy = np.zeros_like(vx)
    vz = np.zeros_like(vx)

    # Assign particles to subdomains
    ions.initialize(x, y, vx, vy, vz)

    def deposit_and_compare(t):
        """
        This function does the following:
        1.  Deposit particle charge onto the grid
        2.  Compute exact charge density
        3.  Plot (and compute difference between) the two charge densities
        """
        # Deposit charge
        sources.deposit(ions)
        sources.current.add_guards()

        # Deposited charge density in the active cells averaged over y
        rho1 = sources.rho.active.mean(axis=0)

        # Exact charge density
        rho2 = 1/euler_prime(lagrange(manifold.x, t), t)

        if plot:
            # Update plot
            lines[0].set_ydata(rho1)
            lines[1].set_ydata(rho2)

        # Root-mean-squared difference between deposited and exact charge
        # density
        return rms(rho1 - rho2)

    # Compute deposited and exact charge density at various times and make sure
    # the difference between the two is never too large
    err = deposit_and_compare(t)

    for i in range(nt):
        ions.drift(dt)
        t += dt
        err = max(err, deposit_and_compare(t))
        if plot:
            plt.pause(1e-7)

    assert err < 1e-2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_burgers(plot=args.plot)
