from skeletor import cppinit, Float, Float2
from skeletor import Grid, Field, Particles, Poisson, Sources
from skeletor.operators.ppic2 import Operators
import numpy
from mpi4py import MPI


def allclose_sorted(a, b, **kwargs):
    """This function first sorts the input arrays 'a' and 'b' (out of place)
    and returns true if the sorted arrays are almost equal to each other
    (element by element)."""
    from numpy import allclose, sort
    return allclose(sort(a), sort(b), **kwargs)


def test_skeletor():

    # Number of grid points in x- and y-direction
    nx, ny = 32, 32

    # Average number of particles per cell
    npc = 256

    # Particle charge and mass
    charge = -1.0
    mass = 1.0

    # Thermal velocity of electrons in x- and y-direction
    vtx, vty = 1.0, 1.0
    # Drift velocity of electrons in x- and y-direction
    vdx, vdy = 0.0, 0.0

    # Timestep
    dt = 0.1
    # Number of timesteps to run for
    nt = 10

    # Synchronize random number generator across ALL processes
    numpy.random.set_state(MPI.COMM_WORLD.bcast(numpy.random.get_state()))

    # Total number of particles in simulation
    np = npc*nx*ny

    # Uniform distribution of particle positions
    x = nx*numpy.random.uniform(size=np).astype(Float)
    y = nx*numpy.random.uniform(size=np).astype(Float)
    # Normal distribution of particle velocities
    vx = vdx + vtx*numpy.random.normal(size=np).astype(Float)
    vy = vdy + vty*numpy.random.normal(size=np).astype(Float)

    global_electrons = []
    global_rho = []

    for comm in [MPI.COMM_SELF, MPI.COMM_WORLD]:

        # Start parallel processing. Calling this function necessary even
        # though `MPI.Init()` has already been called by importing `MPI` from
        # `mpi4py`. The reason is that `cppinit()` sets a number of global
        # variables in the C library source file (`ppic2/pplib2.c`). The
        # returned variables `idproc` and `nvp` are simply the MPI rank (i.e.
        # processor id) and size (i.e. total number of processes),
        # respectively.
        idproc, nvp = cppinit(comm)

        # Create numerical grid. This contains information about the extent of
        # the subdomain assigned to each processor.
        grid = Grid(nx, ny, comm)

        # Maximum number of electrons in each partition
        npmax = int(1.5*np/nvp)

        # Create particle array
        electrons = Particles(npmax, charge, mass)

        # Assign particles to subdomains
        electrons.initialize(x, y, vx, vy, grid)

        # Make sure the numbers of particles in each subdomain add up to the
        # total number of particles
        assert comm.allreduce(electrons.np, op=MPI.SUM) == np

        #######################
        # Test particle drift #
        #######################

        # Set the force to zero (this will of course change in the future).
        fxy = Field(grid, comm, dtype=Float2)
        fxy.fill((0.0, 0.0))

        for it in range(nt):

            # Push particles on each processor. This call also sends and
            # receives particles to and from other processors/subdomains. The
            # latter is the only non-trivial step in the entire code so far.
            electrons.push(fxy, dt)

        # Combine particles from all processes into a single array
        global_electrons.append(
                numpy.concatenate(comm.allgather(electrons[:electrons.np])))

        ##########################
        # Test charge deposition #
        ##########################

        sources = Sources(grid, comm, dtype=Float)
        sources2 = Sources(grid, comm, dtype=Float)

        sources.deposit(electrons)
        sources2.deposit_ppic2(electrons)

        # Make sure the two deposit routines give the same result
        assert numpy.allclose(sources.rho, sources2.rho)
        # Make sure the charge deposited into *all* cells (active plus guard)
        # equals the number of particles times the particle charge
        assert numpy.isclose(sources.rho.sum(), electrons.np*charge)

        # Add charge from guard cells to corresponding active cells.
        # Afterwards erases charge in guard cells.
        sources.rho.add_guards()
        sources2.rho.add_guards_ppic2()

        # Make sure the two add_guards routines give the same result
        assert numpy.allclose(sources.rho, sources2.rho)
        # Make sure the charge deposited into *active* cells (no guard cells)
        # equals the number of particles times the particle charge
        assert numpy.isclose(comm.allreduce(
            sources.rho.trim().sum(), op=MPI.SUM), np*charge)

        # Combine charge density from all processes into a single array
        global_rho += [numpy.concatenate(comm.allgather(sources.rho.trim()))]

        ##########################
        # Compute electric field #
        ##########################

        # Smoothed particle size
        ax, ay = 0.912871, 0.912871

        # Initialize various integro-differential operators
        operators = Operators(grid, ax, ay, np)
        grid.operators = operators

        # Initialize Poisson solver
        poisson = Poisson(grid, ax, ay, np)

        # Solve Gauss's law
        poisson(sources.rho, fxy)
        fxy2 = fxy.copy()

        # Copy data to guard cells from corresponding active cells
        fxy.copy_guards()
        fxy2.copy_guards_ppic2()

        # Make sure the two copy_guards routines give the same result
        assert numpy.allclose(fxy["x"], fxy2["x"])
        assert numpy.allclose(fxy["y"], fxy2["y"])

    # Make sure the the final particle phase space coordinates do not depend
    # on how many processors have been used
    for component in ["x", "y", "vx", "vy"]:
        assert allclose_sorted(
            global_electrons[0][component],
            global_electrons[1][component])

    # The same should be true for the charge density
    assert numpy.allclose(global_rho[0], global_rho[1])


if __name__ == "__main__":
    test_skeletor()
