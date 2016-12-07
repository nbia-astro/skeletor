from skeletor import Float, Float2, Field
from mpi4py.MPI import COMM_WORLD as comm
import numpy
from skeletor.manifolds.mpifft4py import ShearingManifold


def test_shear_gradient_operator(plot=False):
    """This function tests the shear gradient operator implemented with
        mpiFFT4py"""

    # Spatial resolution
    indx, indy = 8, 7
    nx = 1 << indx
    ny = 1 << indy

    ##################################################
    # Solve Ohm's law in shear coordinates with FFTs #
    ##################################################

    # Create numerical grid
    manifold = ShearingManifold(nx, ny, comm)

    # Coordinate arrays
    xx, yy = numpy.meshgrid(manifold.x, manifold.y)

    # alpha parameter in Ohm's law
    alpha = 1

    def rho_analytic(t):
        """Analytic density as a function of time"""
        kx = 2*numpy.pi*ikx/nx
        ky = S*t*kx

        # Initialize density field
        rho = Field(manifold, dtype=Float)
        rho.fill(0.0)

        A = 0.2
        rho[:manifold.nyp, :nx] = 1 + A*numpy.sin(kx*xx + ky*yy)

        return rho

    def E_analytic(t):
        """Analytic electric field as a function of time"""

        # Initialize electric field
        E = Field(manifold, dtype=Float2)
        E.fill((0.0, 0.0))

        kx = 2*numpy.pi*ikx/nx
        ky = S*t*kx

        A = 0.2
        E['x'][:manifold.nyp, :nx] = -alpha*kx*A*numpy.cos(kx*xx+ky*yy) \
            / (1 + A*numpy.sin(kx*xx + ky*yy))
        E['y'][:manifold.nyp, :nx] = -alpha*ky*A*numpy.cos(kx*xx+ky*yy) \
            / (1 + A*numpy.sin(kx*xx + ky*yy))

        return E

    # Initialize electric field
    E = Field(manifold, dtype=Float2)
    E.fill((0.0, 0.0))

    # Rate of shear
    S = -3/2
    # Time step
    dt = 2e-2/abs(S)
    # Amount of time between instances at which the domain is strictly periodic
    tS = manifold.Lx/(abs(S)*manifold.Ly)
    # Start and end time
    tstart = -3*tS
    tend = 3*tS
    # Azimuthal wave number of the ``analytic solution''
    ikx = 1

    # Total number of time steps
    nt = int((tend-tstart)/dt)

    def concatenate(arr):
        """Concatenate local arrays to obtain global arrays
           The result is available on all processors."""
        return numpy.concatenate(comm.allgather(arr))

    # Make initial figure
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.cbook import mplDeprecation
        import warnings

        # Find global solutions
        rho_an = rho_analytic(tstart)
        E_an = E_analytic(tstart)
        global_E_an = concatenate(E_an.trim())
        global_rho_an = concatenate(rho_an.trim())

        if comm.rank == 0:
            plt.rc('image', origin='lower', interpolation='nearest')
            plt.figure(1)
            plt.clf()
            fig, (ax1, ax2, ax3) = plt.subplots(num=1, nrows=3)
            im1 = ax1.imshow(global_rho_an)
            im2 = ax2.imshow(global_E_an["x"])
            im3 = ax3.imshow(global_E_an["y"])
            ax1.set_title(r'$\rho$')
            ax2.set_title(r'$E_x$')
            ax3.set_title(r'$E_y$')
            for ax in (ax1, ax2, ax3):
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')

    for it in range(nt):
        t = tstart + it*dt

        # Density field
        rho = rho_analytic(t)

        # Calculate shear electric field
        manifold.gradient(numpy.log(rho.trim()), E, S*t)
        E['x'] *= -alpha
        E['y'] *= -alpha

        # Calculate analytic field
        E_an = E_analytic(t)

        # Make sure the two solutions are close to each other
        assert numpy.allclose(E_an['x'], E['x'], atol=1e-06)
        assert numpy.allclose(E_an['y'], E['y'], atol=1e-06)

        # Make figures
        if plot:
            if (it % 1 == 0):
                global_rho = concatenate(rho.trim())
                global_E = concatenate(E.trim())
                if comm.rank == 0:
                    im1.set_data(global_rho)
                    im2.set_data(global_E["x"])
                    im3.set_data(global_E["y"])
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                                "ignore", category=mplDeprecation)
                        plt.pause(1e-7)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', '-p', action='store_true')
    args = parser.parse_args()

    test_shear_gradient_operator(plot=args.plot)
