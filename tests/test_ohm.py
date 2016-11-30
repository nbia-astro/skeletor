from skeletor import Float, Float2, Field, Ohm
from skeletor.manifolds.ppic2 import Manifold
from mpi4py.MPI import COMM_WORLD as comm
import numpy


def test_ohm(plot=False):
    """This function tests the solution of Ohm's law, E = -T/m ∇ρ/ρ for a
    simple sinusoidal density profile using the class Ohm and mpiFFT4py.
    The result is compared with the analytical solution using assertions."""

    # Spatial resolution
    indx, indy = 5, 6
    nx = 1 << indx
    ny = 1 << indy

    # Number of particles
    # TODO: Remove because irrelevant for this test
    npc = 1
    np = npc*nx*ny

    #############################################
    # Solve Ohm's law with mpifft4py            #
    #############################################

    # Smoothed particle size
    ax, ay = 0.0, 0.0

    # Create numerical grid
    manifold = Manifold(nx, ny, comm, ax, ay)

    # Initialize Ohm's law solver
    ohm = Ohm(manifold, temperature=1.0, charge=1.0)

    # Coordinate arrays
    x = numpy.arange(manifold.nx, dtype=Float)
    y = manifold.noff + numpy.arange(manifold.nyp, dtype=Float)
    xx, yy = numpy.meshgrid(x, y)

    # Initialize density field
    rho = Field(manifold, dtype=Float)
    rho.fill(0.0)

    # Wavenumbers of mode
    ikx, iky = 1, 2
    kx = 2*numpy.pi*ikx/nx
    ky = 2*numpy.pi*iky/ny

    # Notice that the charge is positive
    A = 0.2
    rho[:manifold.nyp, :nx] = 1 + A*numpy.sin(kx*xx + ky*yy)

    # Initialize electric field
    E = Field(manifold, dtype=Float2)
    E.fill((0.0, 0.0))

    # Solve Ohm's law
    ohm(rho, E, destroy_input=False)
    # Set boundary condition
    E.copy_guards()

    # Calculate the electric field analytically
    factor = -ohm.alpha/rho[:manifold.nyp, :nx]
    Ex = kx*A*numpy.cos(kx*xx + ky*yy)*factor
    Ey = ky*A*numpy.cos(kx*xx + ky*yy)*factor

    # Concatenate local arrays to obtain global arrays
    # The result is available on all processors.
    def concatenate(arr):
        return numpy.concatenate(comm.allgather(arr))

    # Find global solution
    global_E = concatenate(E.trim())
    global_rho = concatenate(rho.trim())

    global_Ex = concatenate(Ex)
    global_Ey = concatenate(Ey)

    # Make sure the two solutions are close to each other
    # This works to machine precision for indx, indy = 3, 4, implying that the
    # truncation error for the fft is below the round-off error. Coarsening (or
    # increasing) the resolution of the grid (try indx, indy = 3, 4) leads to
    # failure. Manually expectecting the difference I find that they are at the
    # 1e-8 level.
    assert numpy.allclose(global_Ex, global_E["x"], atol=1e-6)
    assert numpy.allclose(global_Ey, global_E["y"], atol=1e-6)

    #############
    # Visualize #
    #############

    if plot:
        import matplotlib.pyplot as plt
        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            plt.clf()
            ax1 = plt.subplot2grid((2, 4), (0, 1), colspan=2)
            ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
            ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
            ax1.imshow(global_rho)
            ax2.imshow(global_E["x"])
            ax3.imshow(global_E["y"])

            ax1.set_title(r'$\rho$')
            ax2.set_title(r'$f_x$')
            ax3.set_title(r'$f_y$')
            for ax in (ax1, ax2, ax3):
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
            plt.draw()
            plt.show()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_ohm(plot=args.plot)
