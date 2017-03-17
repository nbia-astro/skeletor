from skeletor import Float3, Field, Particles, Sources
from skeletor import Ohm, InitialCondition
from skeletor.manifolds.second_order import Manifold
import numpy as np
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

    # CFL number
    cfl = 0.5

    # Number of periods to run for
    nperiods = 1

    def rho_an(x, y, t):
        """Analytic density as function of x, y and t"""
        return charge*(1 + A*np.cos(kx*x+ky*y)*np.sin(omega*t))

    def ux_an(x, y, t):
        """Analytic x-velocity as function of x, y and t"""
        return -omega/k*A*np.sin(kx*x+ky*y)*np.cos(omega*t)*kx/k

    def uy_an(x, y, t):
        """Analytic y-velocity as function of x, y and t"""
        return -omega/k*A*np.sin(kx*x+ky*y)*np.cos(omega*t)*ky/k

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm, Lx=1.0, Ly=1.0)

    # x- and y-grid
    xg, yg = np.meshgrid(manifold.x, manifold.y)

    # Sound speed
    cs = np.sqrt(Te/mass)

    # Time step
    dt = cfl/cs*manifold.dx

    # Total number of particles in simulation
    N = npc*nx*ny

    # Wave vector and its modulus
    kx = 2*np.pi*ikx/manifold.Lx
    ky = 2*np.pi*iky/manifold.Ly
    k = np.sqrt(kx*kx + ky*ky)

    # Frequency
    omega = k*cs

    # Simulation time
    tend = 2*np.pi*nperiods/omega

    # Number of time steps
    nt = int(tend/dt)

    # Maximum number of electrons in each partition
    Nmax = int(1.5*N/comm.size)

    # Create particle array
    ions = Particles(manifold, Nmax, charge=charge, mass=mass)

    # Create a uniform density field
    init = InitialCondition(npc, quiet=quiet)
    init(manifold, ions)

    # Particle position in physical units
    x = ions['x']*manifold.dx
    y = ions['y']*manifold.dy

    # Perturbation to particle velocities
    ions['vx'] = ux_an(x, y, t=dt/2)
    ions['vy'] = uy_an(x, y, t=dt/2)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(ions.N, op=MPI.SUM) == N

    # Set the electric field to zero
    E = Field(manifold, dtype=Float3)
    E.fill((0.0, 0.0, 0.0))
    E.copy_guards()
    B = Field(manifold, dtype=Float3)
    B.fill((0.0, 0.0, 0.0))
    B.copy_guards()

    # Initialize sources
    sources = Sources(manifold)

    # Initialize Ohm's law solver
    ohm = Ohm(manifold, temperature=Te, charge=charge)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(ions)
    assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
    sources.current.add_guards()
    sources.current.copy_guards()
    assert np.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), N*charge/npc)

    # Calculate electric field (Solve Ohm's law)
    ohm(sources, B, E)
    # Set boundary condition
    E.copy_guards()

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return np.concatenate(comm.allgather(arr))

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
            vmin, vmax = charge*(1 - A), charge*(1 + A)
            im1 = ax1.imshow(global_rho, vmin=vmin, vmax=vmax)
            im2 = ax2.imshow(global_rho_an, vmin=vmin, vmax=vmax)
            im3 = ax3.plot(xg[0, :], global_rho[0, :], 'b',
                           xg[0, :], global_rho_an[0, :], 'k--')
            ax1.set_title(r'$\rho$')
            ax3.set_ylim(vmin, vmax)
            ax3.set_xlim(0, manifold.Lx)

    t = 0
    diff2 = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        ions.push(E, B, dt)

        # Update time
        t += dt

        # Deposit sources
        sources.deposit(ions)

        # Boundary calls
        sources.current.add_guards()
        sources.current.copy_guards()

        # Calculate forces (Solve Ohm's law)
        ohm(sources, B, E)
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

    # Check if test has passed
    assert np.sqrt(comm.allreduce(diff2, op=MPI.SUM)/nt) < 4e-5*charge


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_ionacoustic(plot=args.plot)
