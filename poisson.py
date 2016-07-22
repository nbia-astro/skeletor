from ppic2_wrapper import cppinit
from ppic2_wrapper import cwpfft2rinit, cppois22, cwppfft2r, cwppfft2r2
from dtypes import Complex, Complex2, Float, Float2, Int
from grid import Grid
from fields import Field
from mpi4py import MPI

import numpy
import matplotlib.pyplot as plt

indx, indy = 5, 5
npc = 256

nx = 1 << indx
ny = 1 << indy

np = nx*ny*npc

comm = MPI.COMM_WORLD

idproc, nvp = cppinit(comm)
# The following definition is surely a Fortran relic
kstrt = idproc + 1

grid = Grid(nx, ny)

x = numpy.arange(grid.nx, dtype=Float)
y = numpy.arange(grid.ny, dtype=Float)

xx, yy = numpy.meshgrid(x, y)

# Array dimensions
nxh = nx//2
nyh = (1 if 1 > ny//2 else ny//2)  # This is just max(1,ny//2)
kxp = (nxh - 1)//nvp + 1
kyp = (ny - 1)//nvp + 1
nye = ny + 2
nxhy = (nxh if nxh > ny else ny)
nxyh = (nx if nx > ny else ny)//2

qe = Field(grid, comm, dtype=Float)
qe.fill(0.0)
fxye = Field(grid, comm, dtype=Float2)
fxye.fill(0.0)

# TODO: Define complex field in dtypes module
qt = numpy.zeros((kxp, nye), Complex)
fxyt = numpy.zeros((kxp, nye), Complex2)

mixup = numpy.zeros(nxhy, Int)
sct = numpy.zeros(nxyh, Complex)
ffc = numpy.zeros((kxp, nyh), Complex)
bs = numpy.zeros((kyp, kxp), Complex2)
br = numpy.zeros((kyp, kxp), Complex2)

# Prepare fft tables
cwpfft2rinit(mixup, sct, indx, indy)

# Smoothed particle size in x/y direction
ax = 0.912871
ay = 0.912871
affp = nx*ny/np

# Calculate form factors
isign = 0
we = cppois22(qt, fxyt, isign, ffc, ax, ay, affp, nx, ny, comm)

# Initialize density field
ikx, iky = 1, 2
qe[:ny, :nx] = numpy.sin(2*numpy.pi*(ikx*xx/nx + iky*yy/ny))

# Transform charge to fourier space with standard procedure:
# updates qt, modifies qe
isign = -1
ttp = cwppfft2r(qe.copy(), qt, bs, br, isign, mixup, sct, indx, indy, comm)

# Calculate force/charge in fourier space with standard procedure:
# updates fxyt, we
isign = -1
we = cppois22(qt, fxyt, isign, ffc, ax, ay, affp, nx, ny, comm)

# Transform force to real space with standard procedure:
# updates fxye, modifies fxyt
isign = 1
cwppfft2r2(fxye, fxyt, bs, br, isign, mixup, sct, indx, indy, comm)

plt.rc('image', origin='lower', interpolation='nearest')
plt.figure(1)
plt.clf()
ax1 = plt.subplot2grid((2, 4), (0, 1), colspan=2)
ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
ax1.imshow(qe)
ax2.imshow(fxye["x"])
ax3.imshow(fxye["y"])
ax1.set_title(r'$\rho$')
ax2.set_title(r'$f_x$')
ax3.set_title(r'$f_y$')
for ax in (ax1, ax2, ax3):
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
plt.draw()
plt.show()
