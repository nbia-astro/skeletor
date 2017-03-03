from types cimport real_t, real2_t, particle_t, grid_t

cdef inline void gather_cic(particle_t particle, real2_t[:,:] F, real2_t *f,
                        grid_t grid, real_t qtmh) nogil

cdef inline void kick(particle_t *particle, real2_t e, real2_t b) nogil

cdef inline void drift2(particle_t *particle, real_t dt, grid_t grid) nogil
