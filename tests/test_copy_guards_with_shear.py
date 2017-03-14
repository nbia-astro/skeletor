from skeletor import Float, Field, ShearField
from skeletor.manifolds.second_order import ShearingManifold
from mpi4py.MPI import COMM_WORLD as comm
import numpy


def test_copy_guards_with_shear(plot=False):
    """This function tests the shearing periodic boundary conditions.
    The result is compared with the analytical solution"""

    # Spatial resolution
    indx, indy = 5, 4
    nx = 1 << indx
    ny = 1 << indy

    # Shear
    S = -3/2
    t = 0.2

    # Create numerical grid
    manifold = ShearingManifold(nx, ny, comm, lbx=1, lby=2, S=S, Omega=0)

    # Coordinate arrays
    xx, yy = numpy.meshgrid(manifold.x, manifold.y)

    # Wavenumbers of mode
    ikx = 1
    kx = 2*numpy.pi*ikx/manifold.Lx
    ky = kx*S*t
    A = 0.2

    # Initialize density field using standard field class
    f = Field(manifold, dtype=Float)
    f.fill(0.0)
    f.active = 1 + A*numpy.sin(kx*xx + ky*yy)

    # Initialize density field using shear field class
    g = ShearField(manifold, time=t, dtype=Float)
    g.active = 1 + A*numpy.sin(kx*xx + ky*yy)

    # Apply boundaries
    f.copy_guards()
    # TODO: Input St can be removed if we let the field know the time and S
    g.copy_guards()

    # Grid including first ghost zone
    xg = (numpy.arange(
            -manifold.lbx, manifold.nx + manifold.lbx) + 0.5)*manifold.dx
    xxg, yyg = numpy.meshgrid(xg, manifold.yg)

    # Analytic field including first ghost-zone
    g_an = 1 + A*numpy.sin(kx*xxg + ky*yyg)

    # Compare field analytic field with field with applied boundary condition
    assert(numpy.allclose(g, g_an))

    if plot:
        if comm.rank == comm.size - 1:
            import matplotlib.pyplot as plt
            plt.rc('image', origin='lower', interpolation='nearest')
            fig, (ax1, ax2) = plt.subplots(num=1, nrows=2)
            ax1.imshow(f)
            ax2.imshow(g)

            plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_copy_guards_with_shear(plot=args.plot)
