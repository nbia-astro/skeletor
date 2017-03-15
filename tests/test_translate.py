from skeletor import Float, ShearField
from skeletor.manifolds.mpifft4py import ShearingManifold
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np


def test_translate(plot=False):
    """This function tests the translation implemented in the ShearField class.
    The result is compared with the analytical solution"""

    # Spatial resolution
    indx, indy = 5, 4
    nx = 1 << indx
    ny = 1 << indy

    # Shear
    S = -3/2
    t = 0.2

    # Create numerical grid
    manifold = ShearingManifold(nx, ny, comm, Lx=2.0, Ly=1.0, lbx=1, lby=2,
                                S=S, Omega=0)

    # Coordinate arrays
    xx, yy = np.meshgrid(manifold.x, manifold.y)

    # Wavenumbers of mode
    ikx = 1
    A = 0.2
    kx = 2*np.pi*ikx/manifold.Lx

    def rho_an(t):
        ky = kx*S*t
        return 1 + A*np.sin(kx*xx + ky*yy)

    # Initialize density field using shear field class
    rho = ShearField(manifold, time=t, dtype=Float)
    rho.fill(0.0)
    rho.active = rho_an(0)
    rho.copy_guards()

    rho.translate(t)

    # Compare field analytic field with field with translated field
    assert(np.allclose(rho_an(t), rho.trim()))

    if plot:
        def concatenate(arr):
            """Concatenate local arrays to obtain global arrays
                The result is available on all processors. """
            return np.concatenate(comm.allgather(arr))

        global_rho = concatenate(rho.trim())
        global_rho_an = concatenate(rho_an(t))
        if comm.rank == 0:
            import matplotlib.pyplot as plt
            plt.rc('image', origin='lower', interpolation='nearest')
            fig, (ax1, ax2) = plt.subplots(num=1, nrows=2)
            ax1.imshow(global_rho_an)
            ax2.imshow(global_rho)

            plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_translate(plot=args.plot)
