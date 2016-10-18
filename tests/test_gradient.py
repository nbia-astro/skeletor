from skeletor import cppinit, Float, Float2, Grid, Field, Operators
from mpi4py.MPI import COMM_WORLD as comm
from mpiFFT4py.line import R2C
from mpi4py import MPI

import numpy


def test_gradient(plot=False):

    # Spatial resolution
    indx, indy = 5, 6
    nx = 1 << indx
    ny = 1 << indy

    # Average number of particles per cell
    npc = 256

    # Smoothed particle size in x/y direction
    ax = 0.912871
    ay = 0.912871

    # Total number of particles
    np = nx*ny*npc

    #############################################
    # Compute gradient with PPIC's parallel FFT #
    #############################################

    # Start parallel processing.
    idproc, nvp = cppinit(comm)

    # Create numerical grid
    grid = Grid(nx, ny, comm)

    # Initialize integro-differential operators
    operators = Operators(grid, ax, ay, np)

    # Coordinate arrays
    x = numpy.arange(grid.nx, dtype=Float)
    y = grid.noff + numpy.arange(grid.nyp, dtype=Float)
    xx, yy = numpy.meshgrid(x, y)

    # Initialize density field
    qe = Field(grid, comm, dtype=Float)
    qe.fill(0.0)
    ikx, iky = 1, 2
    qe[:grid.nyp, :nx] = numpy.sin(2*numpy.pi*(ikx*xx/nx + iky*yy/ny))

    # Initialize force field
    fxye = Field(grid, comm, dtype=Float2)
    fxye.fill((0.0, 0.0))

    # Compute gradient
    operators.gradient(qe, fxye, destroy_input=False)

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
    kx = 2*numpy.pi*numpy.fft.rfftfreq(grid.nx)
    ky = 2*numpy.pi*numpy.fft.fftfreq(grid.ny)
    kx, ky = numpy.meshgrid(kx, ky)

    # Normalization constant
    affp = grid.nx*grid.ny/np

    # Finite size particle shape factor
    s = numpy.exp(-((kx*ax)**2 + (ky*ay)**2)/2)

    # Effective wave numbers
    # TODO: Figure out how the exponential factor is actually derived
    kx_eff = kx*s
    ky_eff = ky*s

    # Transform charge density to Fourier space
    qt = numpy.fft.rfft2(global_qe)

    # Compute gradient in Fourier space and transform back to real space
    fx = affp*numpy.fft.irfft2(1j*kx_eff*qt)
    fy = affp*numpy.fft.irfft2(1j*ky_eff*qt)

    # Make sure Numpy gives the same result
    assert numpy.allclose(fx, global_fxye["x"])
    assert numpy.allclose(fy, global_fxye["y"])

    ###################################
    # Compute gradient with mpiFFT4py #
    ###################################

    # Length vector
    L = numpy.array([ny, nx], dtype=float)
    # Grid size vector
    N = numpy.array([ny, nx], dtype=int)

    # Create FFT object
    FFT = R2C(N, L, MPI.COMM_WORLD, "double")

    # Pre-allocate array for Fourier transform and force
    qe_hat = numpy.zeros(FFT.complex_shape(), dtype=FFT.complex)

    fx_mpi = numpy.zeros_like(qe.trim())
    fy_mpi = numpy.zeros_like(qe.trim())

    # Scaled local wavevector
    k = FFT.get_scaled_local_wavenumbermesh()

    # Define kx and ky (notice that they are swapped due to the grid ordering)
    kx = k[1]
    ky = k[0]

    # Initialize force field
    fxye = Field(grid, comm, dtype=Float2)
    fxye.fill((0.0, 0.0))

    # Normalization constant
    affp = grid.nx*grid.ny/np

    # Finite size particle shape factor
    s = numpy.exp(-((kx*ax)**2 + (ky*ay)**2)/2)

    # Effective wave numbers
    kx_eff = kx*s
    ky_eff = ky*s

    # Transform charge density to Fourier space
    qe_hat = FFT.fft2(qe.trim(), qe_hat)

    # Solve Gauss' law in Fourier space and transform back to real space
    fx_mpi = affp*FFT.ifft2(1j*kx_eff*qe_hat, fx_mpi)
    fy_mpi = affp*FFT.ifft2(1j*ky_eff*qe_hat, fy_mpi)

    # Find global solution
    global_fx_mpi = concatenate(fx_mpi)
    global_fy_mpi = concatenate(fy_mpi)

    # Make sure mpiFFT4py gives the same result
    assert numpy.allclose(global_fx_mpi, global_fxye["x"])
    assert numpy.allclose(global_fy_mpi, global_fxye["y"])

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