from skeletor import Float, Float2, Field
from skeletor.manifolds.mpifft4py import Manifold
from mpi4py.MPI import COMM_WORLD as comm
from mpi4py import MPI

import numpy


def test_gradient(plot=False):

    # Spatial resolution
    indx, indy = 5, 6
    nx = 1 << indx
    ny = 1 << indy

    # Smoothed particle size in x/y direction
    ax = 0.0#0.912871
    ay = 0.0#0.912871

    #############################################
    # Compute gradient with mpifft4py's parallel FFT #
    #############################################

    # Create numerical grid
    manifold = Manifold(nx, ny, comm, ax=ax, ay=ay)

    # Coordinate arrays
    xx, yy = numpy.meshgrid(manifold.x, manifold.y)

    # Initialize density field
    qe = Field(manifold, dtype=Float)
    qe.fill(0.0)
    ikx, iky = 1, 2
    qe.active = numpy.sin(2*numpy.pi*(ikx*xx/nx + iky*yy/ny))

    # Initialize force field
    fxye = Field(manifold, dtype=Float2)
    fxye.fill((0.0, 0.0, 0.0))

    # Compute gradient
    manifold.gradient(qe, fxye)

    # Concatenate local arrays to obtain global arrays (without guard cells).
    # The result is available on all processors.
    def concatenate(arr):
        return numpy.concatenate(comm.allgather(arr))
    global_qe = concatenate(qe.trim())
    global_fxye = concatenate(fxye.trim())

    ##############################################
    # Compute gradient with Numpy's built-in FFT #
    ##############################################

    # Wave number arrays
    kx = 2*numpy.pi*numpy.fft.rfftfreq(manifold.nx)
    ky = 2*numpy.pi*numpy.fft.fftfreq(manifold.ny)
    kx, ky = numpy.meshgrid(kx, ky)

    # Finite size particle shape factor
    s = numpy.exp(-((kx*ax)**2 + (ky*ay)**2)/2)

    # Effective wave numbers
    # TODO: Figure out how the exponential factor is actually derived
    kx_eff = kx*s
    ky_eff = ky*s

    # Transform charge density to Fourier space
    qt = numpy.fft.rfft2(global_qe)

    # Compute gradient in Fourier space and transform back to real space
    fx = numpy.fft.irfft2(1j*kx_eff*qt)
    fy = numpy.fft.irfft2(1j*ky_eff*qt)

    # Make sure Numpy gives the same result
    assert numpy.allclose(fx, global_fxye["x"], atol=1e-6)
    assert numpy.allclose(fy, global_fxye["y"], atol=1e-6)

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
            ax1.imshow(global_qe)
            ax2.imshow(global_fxye["x"])
            ax3.imshow(global_fxye["y"])
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

    test_gradient(plot=args.plot)
