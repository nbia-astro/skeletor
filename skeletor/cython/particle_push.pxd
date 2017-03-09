from types cimport real_t, real3_t, particle_t, grid_t

cdef inline void gather_cic(particle_t particle, real3_t[:,:] F, real3_t *f,
                        real3_t offset) nogil

cdef inline void kick(particle_t *particle, real3_t e, real3_t b) nogil

cdef inline void drift2(particle_t *particle, real_t dt, grid_t grid) nogil

cdef inline void rescale(real3_t *f, real_t qtmh) nogil
