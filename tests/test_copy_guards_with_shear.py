from skeletor import cppinit, Float, Grid, Field, ShearField
from mpi4py.MPI import COMM_WORLD as comm
import numpy


def test_copy_guards_with_shear(plot=False):
    """This function tests the shearing periodic boundary conditions.
    The result is compared with the analytical solution"""

    # Spatial resolution
    indx, indy = 5, 4
    nx = 1 << indx
    ny = 1 << indy

    # Start parallel processing.
    idproc, nvp = cppinit(comm)

    # Create numerical grid
    grid = Grid(nx, ny, comm)

    # Shear
    S = -3/2
    St = 0.2*S

    # Coordinate arrays
    xx, yy = numpy.meshgrid(grid.x, grid.y)

    # Wavenumbers of mode
    ikx = 1
    kx = 2*numpy.pi*ikx/nx
    ky = kx*St
    A = 0.2

    # Initialize density field using standard field class
    f = Field(grid, comm, dtype=Float)
    f.fill(0.0)
    f[:grid.nyp, :nx] = 1 + A*numpy.sin(kx*xx + ky*yy)

    # Initialize density field using shear field class
    g = ShearField(grid, comm, dtype=Float)
    g.fill(0.0)
    g[:grid.nyp, :nx] = 1 + A*numpy.sin(kx*xx + ky*yy)

    # Apply boundaries
    f.copy_guards()
    # TODO: Input St can be removed if we let the field know the time and S
    g.copy_guards(St)

    # Grid including first ghost zone
    xg = numpy.arange(nx+1)
    xxg, yyg = numpy.meshgrid(xg, grid.yg)

    # Analytic field including first ghost-zone
    g_an = 1 + A*numpy.sin(kx*xxg + ky*yyg)

    # Compare field analytic field with field with applied boundary condition
    assert(numpy.allclose(g[:, :(nx+1)], g_an))

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
