from skeletor import Float3, Field, Sources
from skeletor.manifolds.ppic2 import Manifold
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
import matplotlib.pyplot as plt


# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return np.concatenate(comm.allgather(arr))


# Number of grid points in x- and y-direction
nx, ny = 32, 64
# Mean electron charge density
rho0 = 1.0
# Dimensionless amplitude of perturbation
A = 0.1
# Wavenumbers
ikx, iky = 1, 1

# Create numerical grid.
manifold = Manifold(nx, ny, comm, custom_cppois22=True)

# Wave vector and its modulus
kx = 2*np.pi*ikx/manifold.Lx
ky = 2*np.pi*iky/manifold.Ly
k2 = kx*kx + ky*ky

# x- and y-grid
x, y = np.meshgrid(manifold.x, manifold.y)

# Initialize sources
sources = Sources(manifold)
sources.current['t'].active = rho0*(1 + A*np.cos(kx*x + ky*y))
sources.current.copy_guards()

# Compute E = ∇∇⁻²ρ, where ∇⁻² is the inverse Laplacian
E = Field(manifold, dtype=Float3)
manifold.grad_inv_del(sources.rho, E)
E.copy_guards()

# Exact solution
E2 = Field(manifold, dtype=Float3)
E2['x'].active = (kx/k2)*A*np.sin(kx*x + ky*y)
E2['y'].active = (ky/k2)*A*np.sin(kx*x + ky*y)
E2.copy_guards()

# Combine local fields
global_rho = concatenate(sources.rho.trim())
global_E = concatenate(E.trim())
global_E2 = concatenate(E2.trim())


def test_poisson():
    assert np.sqrt(np.mean(global_E['x'] - global_E2['x'])) < 5e-8*(kx/k2)*A
    assert np.sqrt(np.mean(global_E['y'] - global_E2['y'])) < 5e-8*(ky/k2)*A


if __name__ == "__main__":

    test_poisson()

    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(global_E['x'][0, :], 'k')
    plt.plot(global_E2['x'][0, :], 'r--')
    plt.subplot(212)
    plt.plot(global_E['y'][0, :], 'k')
    plt.plot(global_E2['y'][0, :], 'r--')
    plt.show()
