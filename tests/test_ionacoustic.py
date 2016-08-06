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

    # x-grid
    xg = numpy.arange(0,nx)

    # Average number of particles per cell
    npc = 256

    # Particle charge and mass
    charge = 1.0
    mass = 1.0

    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 0.0, 0.0
    # Velocity perturbation of ions in x- and y-direction
    vdx, vdy = 1e-3, 1e-3

    # Timestep
    dt = 0.1
    # Number of timesteps to run for
    nt = 250

    # Total number of particles in simulation
    np = npc*nx*ny

    # Wavenumbers
    kx = 2*numpy.pi/nx
    ky = 0#2*numpy.pi/ny
    k  = numpy.sqrt((kx**2+ky**2))

    # Frequency (TODO: Introduce cs and dependence on ky)
    omega = kx

    # Analytic density (TODO: Only works for ky = 0, write down 2D solution)
    def rho_an(t):
        return 1 - vdx*numpy.cos(kx*xg)*numpy.sin(omega*t)

    if quiet:
        # Uniform distribution of particle positions (quiet start)
        assert(numpy.sqrt(npc) % 1 == 0)
        dx = 1/int(numpy.sqrt(npc))
        dy = dx
        X = numpy.arange(0, nx, dx)
        Y = numpy.arange(0, ny, dy)
        x, y = numpy.meshgrid(X, Y)
        x = x.flatten()
        y = y.flatten()
    else:
        x = nx*numpy.random.uniform(size=np).astype(Float)
        y = ny*numpy.random.uniform(size=np).astype(Float)

    # Perturbation to particle velocities
    vx = vdx*numpy.sin(kx*x + ky*y)*kx/k
    vy = vdy*numpy.sin(kx*x + ky*y)*ky/k

    # Add thermal velocity
    vx += vtx*numpy.random.normal(size=np).astype(Float)
    vy += vty*numpy.random.normal(size=np).astype(Float)

    # Start parallel processing
    idproc, nvp = cppinit(comm)

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

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return numpy.concatenate(comm.allgather(arr))

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        global_rho = concatenate(sources.rho.trim())
        global_E = concatenate(E.trim())

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            fig, (ax1, ax2) = plt.subplots (num=1, ncols=2)
            im1 = ax1.imshow(global_rho,vmin=1-vdx,vmax=1+vdx)
            im2 = ax2.plot(xg, global_rho[0,:],'b', xg, rho_an(0),'k--')
            ax1.set_title(r'$\rho$')
            ax2.set_ylim(1-vdx,1+vdx)
            ax2.set_xlim(0,x[-1])

    t = 0
    ##########################################################################
    # Main loop over time                                                    #
    ##########################################################################
    for it in range(nt):
        # Calculate force from electric field
        fxy['x'] = E['x']*charge
        fxy['y'] = E['y']*charge
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        ions.push(fxy, dt)

        # Update time
        t += dt

        # Deposit sources
        sources.deposit_ppic2(ions)
        # Adjust density (TODO: we should do this somewhere else)
        sources.rho /= npc
        assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)
        # Boundary calls
        sources.rho.add_guards_ppic2()
        sources.rho.copy_guards_ppic2()

        assert numpy.isclose(comm.allreduce(
            sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

        # Calculate forces (Solve Ohm's law)
        ohm(sources.rho, E, destroy_input=False)

        # Make figures
        if plot:
            if (it % 4 == 0):
                global_rho = concatenate(sources.rho.trim())
                global_E = concatenate(E.trim())
                if comm.rank == 0:
                    im1.set_data(global_rho)
                    im2[0].set_ydata(global_rho[0,:])
                    im2[1].set_ydata(rho_an(t))
                    plt.pause(1e-7)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_ionacoustic(plot=args.plot)
