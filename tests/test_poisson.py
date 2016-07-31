from skeletor import cppinit, Float, Float2, Grid, Field, Poisson
from mpi4py.MPI import COMM_WORLD as comm

import numpy
import matplotlib.pyplot as plt

# Spatial resolution
indx, indy = 5, 5
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
# Solve Gauss' law with PPIC's parallel FFT #
#############################################

# Start parallel processing.
idproc, nvp = cppinit(comm)

# Create numerical grid
grid = Grid(nx, ny, comm)

# Initialize Poisson solver
poisson = Poisson(grid, ax, ay, np)

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

# Solve Gauss' law
poisson(qe, fxye, destroy_input=False)

##############################################
# Solve Gauss' law with Numpy's built-in FFT #
##############################################

# Concatenate local arrays to obtain global arrays (without guard cells).
# The result is available on all processors.
def concatenate(arr):
    return numpy.concatenate(comm.allgather(arr))
global_qe = concatenate(qe.trim())
global_fxye = concatenate(fxye.trim())

# Wave number arrays
kx = 2*numpy.pi*numpy.fft.rfftfreq(grid.nx)
ky = 2*numpy.pi*numpy.fft.fftfreq(grid.ny)
kx, ky = numpy.meshgrid(kx, ky)

# Normalization constant
affp = grid.nx*grid.ny/np

# Compute inverse wave number squared
k2 = kx**2 + ky**2
k2[0, 0] = 1.0
k21 = 1.0/k2
k21[0, 0] = 0.0
k2[0, 0] = 0.0

# Effective inverse wave number for finite size particles
# TODO: Figure out how the exponential factor is actually derived
k21_eff = k21*numpy.exp(-((kx*ax)**2 + (ky*ay)**2))

# Transform charge density to Fourier space
qt = numpy.fft.rfft2(global_qe)

# Solve Gauss' law in Fourier space and transform back to real space
fx = affp*numpy.fft.irfft2(-1j*kx*k21_eff*qt)
fy = affp*numpy.fft.irfft2(-1j*ky*k21_eff*qt)

# Make sure the two solutions are close to each other
assert numpy.allclose(fx, global_fxye["x"])
assert numpy.allclose(fy, global_fxye["y"])

#############
# Visualize #
#############

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
