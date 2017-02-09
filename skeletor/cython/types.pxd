cimport mpi4py.MPI as MPI

# Make sure these definitions match those in "picksc/ppic2/precision.h"
ctypedef double real_t
ctypedef double complex complex_t

cdef struct particle_t:
    real_t x, y, vx, vy

cdef struct real2_t:
    real_t x, y

cdef struct complex2_t:
    complex_t x, y

cdef class grid_t(object):
    """Grid extension type.
    This is inherited by the Grid class (see grid.py)."""
    cdef public int nx, ny
    cdef public MPI.Comm comm
    cdef public real_t edges[2]
    cdef public int nyp, noff, nypmx, nypmn
