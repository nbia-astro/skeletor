from skeletor.manifolds.second_order import Manifold
from skeletor.particles import Particles
from skeletor.initial_condition import DensityPertubation
from skeletor.sources import Sources
from mpi4py.MPI import COMM_WORLD as comm, SUM
import numpy as np

# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 128, 8
# Box size in x- and y-direction (square cells!)
Lx, Ly = 4, 1
# Coordinate origin
x0 = -Lx/2
y0 = -Ly/2
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
manifold = Manifold(nx, ny, comm, Lx=Lx, Ly=Ly, x0=x0, y0=y0)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass)

# Initialize sources
sources = Sources(manifold)


def perturb_and_deposit(quiet, plot, num):

    # Make random number generation predictable
    # We do this so that the last assertion test below never fails
    if not quiet:
        np.random.seed(0)

    # Create a uniform density field
    perturbation = DensityPertubation(npc, ikx, iky, ampl,
                                      quiet=quiet, global_init=True)
    # init(manifold, ions)
    perturbation(manifold, ions)

    # Deposit sources
    sources.deposit(ions)
    assert np.isclose(sources.rho.sum(), ions.N*charge/npc)
    sources.add_guards()
    sources.copy_guards()
    assert np.isclose(comm.allreduce(
        sources.rho.trim().sum(), op=SUM), N*charge/npc)

    def concatenate(arr):
        return np.concatenate(comm.allgather(arr))

    rho = np.concatenate(comm.allgather(sources.rho.trim())).mean(axis=0)
    rho_exact = ions.n0*ions.charge*(1 + ampl*np.cos(kx*(manifold.x - x0)))

    if quiet:
        assert np.sqrt(np.mean((rho - rho_exact)**2)) < 0.005*ampl
    else:
        ampl_fft = 2*abs(np.fft.rfft(rho)[1])/manifold.nx
        assert abs(ampl_fft - ampl) < 0.02*ampl

    if plot and comm.rank == 0:
        import matplotlib.pyplot as plt
        plt.figure(num)
        plt.clf()
        plt.plot(rho, 'k', rho_exact, 'r--')


def test_density_perturbation_quiet(plot=False):
    quiet = True
    num = 1
    perturb_and_deposit(quiet, plot, num)


def test_density_perturbation_noisy(plot=False):
    quiet = False
    num = 2
    perturb_and_deposit(quiet, plot, num)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_density_perturbation_quiet(plot=args.plot)
    test_density_perturbation_noisy(plot=args.plot)

    if args.plot and comm.rank == 0:
        import matplotlib.pyplot as plt
        plt.draw()
        plt.show()
