from skeletor import Float, Float3
from skeletor import Field, Particles, Ohm, Sources
from skeletor.manifolds.second_order import Manifold
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
    # Domain size
    Lx, Ly = 1.0, 1.0

    # Average number of particles per cell
    npc = 256

    # Particle charge and mass
    charge = 1.0
    mass = 1.0

    # Thermal velocity of electrons in x, y and z-direction
    vtx, vty, vtz = 1.0, 1.0, 1.0
    # Drift velocity of electrons in x, y and z-direction
    vdx, vdy, vdz = 0.0, 0.0, 0.0

    # Timestep
    dt = 0.1
    # Number of timesteps to run for
    nt = 10

    # Synchronize random number generator across ALL processes
    numpy.random.set_state(MPI.COMM_WORLD.bcast(numpy.random.get_state()))

    # Total number of particles in simulation
    np = npc*nx*ny

    # Uniform distribution of particle positions
    x = Lx*numpy.random.uniform(size=np).astype(Float)
    y = Ly*numpy.random.uniform(size=np).astype(Float)
    # Normal distribution of particle velocities
    vx = vdx + vtx*numpy.random.normal(size=np).astype(Float)
    vy = vdy + vty*numpy.random.normal(size=np).astype(Float)
    vz = vdz + vtz*numpy.random.normal(size=np).astype(Float)

    global_ions = []
    global_rho = []

    for comm in [MPI.COMM_SELF, MPI.COMM_WORLD]:

        # Create numerical grid. This contains information about the extent of
        # the subdomain assigned to each processor.
        manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly)

        # Maximum number of ions in each partition
        npmax = int(2.0*np/comm.size)

        # Create particle array
        ions = Particles(manifold, npmax, charge=charge, mass=mass)

        # Assign particles to subdomains
        ions.initialize(x, y, vx, vy, vz)

        # Make sure the numbers of particles in each subdomain add up to the
        # total number of particles
        assert comm.allreduce(ions.np, op=MPI.SUM) == np

        #######################
        # Test particle drift #
        #######################

        # Set the force to zero (this will of course change in the future).
        E = Field(manifold, dtype=Float3)
        E.fill((0.0, 0.0, 0.0))
        E.copy_guards()

        B = Field(manifold, dtype=Float3)
        B.fill((0.0, 0.0, 0.0))
        B.copy_guards()

        for it in range(nt):

            # Push particles on each processor. This call also sends and
            # receives particles to and from other processors/subdomains. The
            # latter is the only non-trivial step in the entire code so far.
            ions.push(E, B, dt)

        # Combine particles from all processes into a single array
        global_ions.append(
                numpy.concatenate(comm.allgather(ions[:ions.np])))

        ##########################
        # Test charge deposition #
        ##########################

        sources = Sources(manifold, npc)

        sources.deposit(ions)

        # Make sure the charge deposited into *all* cells (active plus guard)
        # equals the number of particles times the particle charge
        assert numpy.isclose(sources.rho.sum(), ions.np*charge/npc)

        # Add charge from guard cells to corresponding active cells.
        # Afterwards erases charge in guard cells.
        sources.rho.add_guards()
        sources.J.add_guards_vector()

        # Make sure the charge deposited into *active* cells (no guard cells)
        # equals the number of particles times the particle charge
        assert numpy.isclose(comm.allreduce(
            sources.rho.trim().sum(), op=MPI.SUM), np*charge/npc)

        sources.rho.copy_guards()
        sources.J.copy_guards()

        # Combine charge density from all processes into a single array
        global_rho += [numpy.concatenate(comm.allgather(sources.rho.trim()))]

        ##########################
        # Compute electric field #
        ##########################

        # Initialize Ohm solver
        ohm = Ohm(manifold)

        # Solve Gauss's law
        ohm(sources, B, E)

        # Copy data to guard cells from corresponding active cells
        E.copy_guards()

        # TODO: Previously the solution of Gauss' law obtained above was
        # compared to the solution obtained by PPIC2, assuming that the latter
        # is correct. Now we should probably make sure that our numerical
        # solution is reasonably close to the exact solutiion.

    # Make sure the the final particle phase space coordinates do not depend
    # on how many processors have been used
    for component in ["x", "y", "vx", "vy"]:
        assert allclose_sorted(
            global_ions[0][component],
            global_ions[1][component])

    # The same should be true for the charge density
    assert numpy.allclose(global_rho[0], global_rho[1])


if __name__ == "__main__":
    test_skeletor()
