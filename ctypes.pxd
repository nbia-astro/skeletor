ctypedef float float_t
ctypedef float complex complex_t

cdef struct particle_t:
    float_t x, y, vx, vy

cdef struct force_t:
    float_t x, y
