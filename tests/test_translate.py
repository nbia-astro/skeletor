from skeletor import cppinit, Float, Grid, ShearField
from mpi4py.MPI import COMM_WORLD as comm
import numpy


def test_translate(plot=False):
    """This function tests the translation implemented in the ShearField class.
    The result is compared with the analytical solution"""

    # Spatial resolution
    indx, indy = 5, 4
    nx = 1 << indx
    ny = 1 << indy

    # Start parallel processing.
    idproc, nvp = cppinit(comm)

    # Create numerical grid
    grid = Grid(nx, ny, comm, nlbx=1, nubx=2, nlby=3, nuby=4)

    # Shear
    S = -3/2
    t = 0.2

    # Coordinate arrays
    xx, yy = numpy.meshgrid(grid.x, grid.y)

    # Wavenumbers of mode
    ikx = 1
    A = 0.2
    kx = 2*numpy.pi*ikx/nx

    def rho_an(t):
        ky = kx*S*t
        return 1 + A*numpy.sin(kx*xx + ky*yy)

    # Initialize density field using shear field class
    rho = ShearField(grid, dtype=Float)
    rho.fill(0.0)
    rho.active = rho_an(0)
    rho.copy_guards(0)

    rho.translate(S*t)

    # Compare field analytic field with field with translated field
    assert(numpy.allclose(rho_an(t), rho.trim()))

    if plot:
        def concatenate(arr):
            """Concatenate local arrays to obtain global arrays
                The result is available on all processors. """
            return numpy.concatenate(comm.allgather(arr))

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
