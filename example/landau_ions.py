from skeletor import cppinit, Float, Float3, Field, Particles, Sources
from skeletor import Ohm
from skeletor.manifolds.second_order import Manifold
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
    charge = 1.0
    mass = 1.0
    # Electron temperature
    Te = 1.0
    # Ion temperature
    Ti = 1/5
    # Dimensionless amplitude of perturbation
    A = 0.01
    # Wavenumbers
    ikx, iky = 1, 0
    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = numpy.sqrt(Ti/mass), 0.0
    # CFL number
    cfl = 0.5
    # Number of periods to run for
    nperiods = 3.0

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
    vz = numpy.zeros_like(vx)

    # Add thermal velocity
    vx += vtx*numpy.random.normal(size=np).astype(Float)
    vy += vty*numpy.random.normal(size=np).astype(Float)

    # Start parallel processing
    idproc, nvp = cppinit(comm)

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    manifold = Manifold(nx, ny, comm)

    # x- and y-grid
    xg, yg = numpy.meshgrid(manifold.x, manifold.y)

    # Pair of Fourier basis functions with the specified wave numbers.
    # The basis functions are normalized so that the Fourier amplitude can be
    # computed by summing rather than averaging.
    S = numpy.sin(kx*xg + ky*yg)/(nx*ny)
    C = numpy.cos(kx*xg + ky*yg)/(nx*ny)

    # Maximum number of electrons in each partition
    npmax = int(1.5*np/nvp)

    # Create particle array
    ions = Particles(manifold, npmax, charge=charge, mass=mass)

    # Assign particles to subdomains
    ions.initialize(x, y, vx, vy, vz)

    # Make sure the numbers of particles in each subdomain add up to the
    # total number of particles
    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Set the electric field to zero
    E = Field(manifold, comm, dtype=Float3)
    E.fill((0.0, 0.0, 0.0))
    E.copy_guards()

    B = Field(manifold, comm, dtype=Float3)
    B.fill((0.0, 0.0, 0.0))
    B.copy_guards()

    # Initialize sources
    sources = Sources(manifold)

    # Initialize Ohm's law solver
    ohm = Ohm(manifold, temperature=Te, charge=charge)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(ions)
    assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
    sources.current.add_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)
    sources.current.copy_guards()

    # Calculate electric field (Solve Ohm's law)
    ohm(sources, B, E)
    # Set boundary condition
    E.copy_guards()

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
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.rc('image', aspect='auto')
            plt.figure(1)
            plt.clf()
            fig, (ax1, ax2, ax3) = plt.subplots(num=1, nrows=3)
            im1 = ax1.imshow(global_rho)
            im2 = ax2.imshow(global_E['x'])
            im3 = ax3.imshow(global_E['y'])
            ax1.set_title(r'$\rho$')
            ax2.set_title(r'$E_x$')
            ax3.set_title(r'$E_y$')

    # Compute square of Fourier amplitude by projecting the local density
    ampl2 = []
    time = []

    # Initial time
    t = 0

    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        ions.push(E, B, dt)

        # Update time
        t += dt
        time += [t]

        # Deposit sources
        sources.deposit(ions)

        # Boundary calls
        sources.current.add_guards()
        sources.current.copy_guards()

        # Calculate forces (Solve Ohm's law)
        ohm(sources, B, E)
        # Set boundary condition
        E.copy_guards()

        # Compute square of Fourier amplitude by projecting the local density
        # onto the local Fourier basis
        rho = sources.rho.trim()
        ampl2 += [(S*rho).sum()**2 + (C*rho).sum()**2]

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

    # Sum squared amplitude over processor, then take the square root
    ampl = numpy.sqrt(comm.allreduce(ampl2, op=MPI.SUM))
    # Convert list of times into NumPy array
    time = numpy.array(time)

    # Test if growth rate is correct
    if comm.rank == 0:
        from scipy.signal import argrelextrema

        # Find first local maximum
        index = argrelextrema(ampl, numpy.greater)
        tmax = time[index][0]
        ymax = ampl[index][0]

        # Theoretical gamma (TODO: Solve dispersion relation here)
        gamma_t = -0.0203927225606

        if plot or fitplot:
            import matplotlib.pyplot as plt
            # Create figure
            plt.figure(2)
            plt.clf()
            plt.semilogy(time, ampl, 'b')
            plt.semilogy(time, ymax*numpy.exp(gamma_t*(time - tmax)), 'k-')

            plt.title(r'$|\hat{\rho}(ikx=%d, iky=%d)|$' % (ikx, iky))
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
