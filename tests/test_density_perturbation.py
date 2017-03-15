from skeletor.manifolds.second_order import Manifold
from skeletor.particles import Particles
from skeletor.initial_condition import InitialCondition, DensityPertubation
from skeletor.sources import Sources
from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy as np

# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 32, 8
# Grid size in x- and y-direction (square cells!)
Lx = 4
Ly = 1
dx = Lx/nx
# Average number of particles per cell
npc = 256
# Particle charge and mass
charge = 1.0
mass = 1.0
# Dimensionless amplitude of perturbation
ampl = 0.1
# Wavenumbers
ikx, iky = 1, 0

# Total number of particles in simulation
N = npc*nx*ny

# Wave vector and its modulus
kx = 2*np.pi*ikx/Lx
ky = 2*np.pi*iky/Ly

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass)

# Create a uniform density field
init = InitialCondition(npc, quiet=quiet)
perturbation = DensityPertubation(npc, ikx, iky, ampl)
# init(manifold, ions)
perturbation(manifold, ions)

# Initialize sources
sources = Sources(manifold)


def test_density_perturbation(plot=False):

    # Deposit sources
    sources.deposit(ions)
    assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
    sources.current.add_guards()
    sources.current.copy_guards()
    assert np.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=SUM), N*charge/npc)

    def concatenate(arr):
        return np.concatenate(comm.allgather(arr))

    rho = np.concatenate(comm.allgather(sources.rho.trim())).mean(axis=0)
    rho_exact = ions.n0*ions.charge*(1 + ampl*np.cos(kx*manifold.x))

    assert np.sqrt(np.mean((rho - rho_exact)**2)) < 1e-3

    if plot and comm.rank == 0:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        plt.plot(rho, 'k', rho_exact, 'r--')
        plt.draw()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_density_perturbation(plot=args.plot)
