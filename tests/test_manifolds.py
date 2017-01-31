from skeletor import Float, Float2, Field
# from skeletor.manifolds.ppic2 import Manifold as ppic2
from skeletor.manifolds.mpifft4py import Manifold as mpifft4py
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
    E = Field(manifold, dtype=Float2)
    manifold.gradient(rho, E)
    # Return components of the gradient in the active cells
    return E.active['x'], E.active['y']


def test_manifolds():
    """Compute gradient of a monochromatic density field on various manifolds
    and compare with exact result."""
    # TODO: Split this up into individual tests for each manifold
    # TODO: Do the same with Poisson's equation (grad_inv_del)

    # Spatial resolution
    indx, indy = 5, 6
    nx = 1 << indx
    ny = 1 << indy

    # Coordinate arrays
    nyp = ny//comm.size
    noff = nyp*comm.rank
    x = np.arange(nx, dtype=Float)
    y = noff + np.arange(nyp, dtype=Float)
    xx, yy = np.meshgrid(x, y)

    # Wave numbers and phase
    ikx, iky = 1, 2
    kx = 2*np.pi*ikx/nx
    ky = 2*np.pi*iky/ny
    phase = kx*xx + ky*yy

    # Error tolerance for single and double precision
    # eps_single = 2e-7
    eps_double = 1e-15

    # PPIC2 manifold
    # FIXME: How come this is effectively single precision? Didn't we
    # generalize PPIC2's routines?
    # manifold = ppic2(nx, ny, comm)
    # Ex, Ey = gradient(np.sin(phase), manifold)
    # assert rms_diff(Ex, kx*np.cos(phase)) < eps_single
    # assert rms_diff(Ey, ky*np.cos(phase)) < eps_single

    # mpiFFT4py manifold
    manifold = mpifft4py(nx, ny, comm)
    Ex, Ey = gradient(np.sin(phase), manifold)
    assert rms_diff(Ex, kx*np.cos(phase)) < eps_double
    assert rms_diff(Ey, ky*np.cos(phase)) < eps_double

    # Second-order finite-difference manifold
    manifold = second_order(nx, ny, comm)
    Ex, Ey = gradient(np.sin(phase), manifold)
    # Note the appearance of effective wave numbers. This accounts for the
    # discretization error.
    assert rms_diff(Ex, np.sin(kx)*np.cos(phase)) < eps_double
    assert rms_diff(Ey, np.sin(ky)*np.cos(phase)) < eps_double


if __name__ == "__main__":

    test_manifolds()
