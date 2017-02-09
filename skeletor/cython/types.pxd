# Make sure these definitions match those in "picksc/ppic2/precision.h"
ctypedef double real_t
ctypedef double complex complex_t

cdef struct particle_t:
    real_t x, y, vx, vy

cdef struct real2_t:
    real_t x, y

cdef struct complex2_t:
    complex_t x, y
