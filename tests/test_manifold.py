from skeletor import Float, Float3, Field
# from skeletor.manifolds.ppic2 import Manifold as ppic2
# from skeletor.manifolds.mpifft4py import Manifold as mpifft4py
from skeletor.manifolds.second_order import Manifold as second_order
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np


def rms_diff(a, b):
    """Root-mean-squared difference between two arrays."""
    return np.sqrt(((a - b)**2).mean())


def gradient(rho_active, manifold):
    """Compute gradient of a density field on a given manifold.
    `rho_active` is the density field w/o guard cells."""
    # Define density field w/ guard cells and set its active cells equal
    # to the given input
    rho = Field(manifold, dtype=Float)
    rho.active = rho_active
    # Set boundary condition
    rho.copy_guards()
    # Compute gradient
    E = Field(manifold, dtype=Float3)
    manifold.gradient(rho, E)
    # Return components of the gradient in the active cells
    return E.active['x'], E.active['y']


def test_manifold():
    """Compute gradient of a monochromatic density field on with the second
    order manifold and compare with exact result."""
    # TODO: Do the same with Poisson's equation (grad_inv_del)
    # TODO: Do the same for the curl and the interpolation operator

    # Spatial resolution
    indx, indy = 5, 6
    nx = 1 << indx
    ny = 1 << indy
    # Setting dx, dy not equal to 1 reduces the accuracy to around 1e-14.
    Lx, Ly = nx, ny

    # Wave numbers and phase
    ikx, iky = 1, 2
    kx = 2*np.pi*ikx/Lx
    ky = 2*np.pi*iky/Ly

    # Error tolerance for single and double precision
    eps_single = 2e-7
    eps_double = 1e-15

    # Second-order finite-difference manifold
    manifold = second_order(nx, ny, comm, Lx=Lx, Ly=Ly)
    # x- and y-grid
    xx, yy = np.meshgrid(manifold.x, manifold.y)
    phase = kx*xx + ky*yy
    Ex, Ey = gradient(np.sin(phase), manifold)
    # Note the appearance of effective wave numbers. This accounts for the
    # discretization error.
    dx = manifold.dx
    dy = manifold.dy
    kx_eff = np.sin(kx*dx)/dx
    ky_eff = np.sin(ky*dy)/dy
    assert rms_diff(Ex, kx_eff*np.cos(phase)) < eps_double
    assert rms_diff(Ey, ky_eff*np.cos(phase)) < eps_double

if __name__ == "__main__":

    test_manifold()
