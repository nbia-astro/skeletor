from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
from skeletor import Ohm
import numpy
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
    charge = 1.0
    mass = 1.0

    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 0.0, 0.0
    # Velocity perturbation of ions in x- and y-direction
    vdx, vdy = 0.001, 0.001

    # Timestep
    dt = 0.1
    # Number of timesteps to run for
    nt = 50

    # Total number of particles in simulation
    np = npc*nx*ny

    # Wavenumbers
    kx = 2*numpy.pi/nx
    ky = 2*numpy.pi/ny

    if quiet:
        # Uniform distribution of particle positions (quiet start)
        dx = 1/int(numpy.sqrt(npc))
        dy = dx
        X = numpy.arange(0,nx,dx)
        Y = numpy.arange(0,ny,dy)
        x, y = numpy.meshgrid(X, Y)
        x = x.flatten()
        y = y.flatten()
    else:
        x = nx*numpy.random.uniform(size=np).astype(Float)
        y = ny*numpy.random.uniform(size=np).astype(Float)

    # Normal distribution of particle velocities
    vx = vdx*numpy.sin(kx*x) + vtx*numpy.random.normal(size=np).astype(Float)
    vy = vdy*numpy.sin(ky*y) + vty*numpy.random.normal(size=np).astype(Float)

    # Start parallel processing
    idproc, nvp = cppinit(comm)

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return numpy.concatenate(comm.allgather(arr))

    # Create numerical grid. This contains information about the extent of
    # the subdomain assigned to each processor.
    grid = Grid(nx, ny, comm)

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
    E.fill(0.0)

    # Set force field to zero
    fxy = Field(grid, comm, dtype=Float2)
    fxy.fill(0.0)

    # Initialize sources
    sources = Sources(grid, comm, dtype=Float)

    # Initialize Ohm's law solver
    ohm = Ohm(grid, temperature=1.0, charge=charge)

    # Calculate initial density and force

    # Deposit sources
    sources.deposit(ions)
    # Adjust density (we should do this somewhere else)
    sources.rho /= npc
    assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
    sources.rho.add_guards()
    sources.rho.copy_guards()
    assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

    # Calculate electric field (Solve Ohm's law)
    ohm(sources.rho, E, destroy_input=False)

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        global_rho = concatenate(sources.rho.trim())
        global_E   = concatenate(E.trim())
        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            plt.clf()
            ax1 = plt.subplot2grid((2, 4), (0, 1), colspan=2)
            ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
            ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
            im1 = ax1.imshow(global_rho)
            im2 = ax2.imshow(global_E["x"])
            im3 = ax3.imshow(global_E["y"])
            ax1.set_title(r'$\rho$')
            ax2.set_title(r'$f_x$')
            ax3.set_title(r'$f_y$')
            for ax in (ax1, ax2, ax3):
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
            plt.savefig("test.{:04d}.png".format(0))

    # Main loop over time
    for it in range(nt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains. The
        # latter is the only non-trivial step in the entire code so far.
        fxy['x'] = E['x']*charge
        fxy['y'] = E['y']*charge
        ions.push(fxy, dt)

        # Deposit sources
        sources.deposit(ions)
        # Adjust density (we should do this somewhere else)
        sources.rho /= npc
        assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
        sources.rho.add_guards()
        sources.rho.copy_guards()

        assert numpy.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

        # Calculate forces (Solve Ohm's law)
        ohm(sources.rho, E, destroy_input=False)

        # Make figures
        if plot:
            import matplotlib.pyplot as plt
            global_rho = concatenate(sources.rho.trim())
            global_E = concatenate(E.trim())
            if comm.rank == 0:
                im1.set_data(global_rho)
                im2.set_data(global_E["x"])
                im3.set_data(global_E["y"])
                im1.autoscale()
                im2.autoscale()
                im3.autoscale()
                plt.savefig("test.{:04d}.png".format(it))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_ionacoustic(plot=args.plot)

    # Code produces a ton of png files (sorry! I really tried to get animation
    # to work but I failed...). Combine into a movie by running
    # ffmpeg -framerate 20 -i test.%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    # rm *.png
