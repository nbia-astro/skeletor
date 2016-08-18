from skeletor import cppinit, Float, Float2, Grid, Field, Poisson
from mpi4py.MPI import COMM_WORLD as comm
import numpy


def isclose_cmplx(a, b, **kwds):
    return numpy.logical_and(
            numpy.isclose(a.real, b.real, **kwds),
            numpy.isclose(a.imag, b.imag, **kwds))


def diff_cmplx(a, b):
    return numpy.sqrt((a.real - b.real)**2 + (a.imag - b.imag)**2)


# Spatial resolution
indx, indy = 6, 4
nx = 1 << indx
ny = 1 << indy

# Average number of particles per cell
npc = 256

# Smoothed particle size in x/y direction
ax = 0.912871
ay = 0.912871

# Total number of particles
np = nx*ny*npc

# Normalization constant
affp = 1.0/npc

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

# Initialize force field
fxye = Field(grid, comm, dtype=Float2)
fxye.fill((0.0, 0.0))

# Fill charge density with random numbers. Normalization is such that PPIC2's
# FFT gives back amplitudes of order unity.
qe[:grid.nyp, :nx] = numpy.random.rand(grid.nyp, nx)*nx*grid.nyp
# Get FFT of charge density as byproduct from solving Poisson's equation.
ttp, we, qt, fxyt = poisson(qe, fxye, destroy_input=False)

# Use Numpy's FFT routines to compute transformed charge density
qt2 = numpy.zeros_like(qt)
norm = 1.0/(nx*ny)
qt_np_tr = norm*numpy.fft.rfft2(qe[:grid.nyp, :nx]).transpose()

# PPIC2's and Numpy's FFT routines use different formats for the transformed
# data. It seems PPIC2 uses the "Perm" format whereas Numpy uses the "CCS"
# format, see https://software.intel.com/en-us/node/502293
qt2[:, :-2] = qt_np_tr[:-1, :]
qt2[0, 0] += 1j*qt_np_tr[-1, 0].real
qt2[0, ny//2] += 1j*qt_np_tr[-1, ny//2].real
for iky in range(ny//2 + 1, ny):
    qt2[0, iky] = qt_np_tr[-1, iky]

# Make sure that modulo differences in the format used, the two FFTs agree
kxp, nye = qt.shape
for ikx in range(kxp):
    for iky in range(ny):
        a = qt[ikx, iky]
        b = qt2[ikx, iky]
        if not isclose_cmplx(a, b, atol=1e-5):
            msg = "ikx = {}, iky = {}, diff = {}"
            print(msg.format(ikx, iky, diff_cmplx(a, b)))
