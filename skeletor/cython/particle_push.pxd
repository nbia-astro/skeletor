from types cimport real_t, real2_t, real3_t, particle_t, grid_t

cdef inline void gather_cic(particle_t particle, real3_t[:,:] F, real3_t *f,
                        real2_t offset) nogil

cdef inline void kick_particle(particle_t *particle,
                               real3_t e, real3_t b) nogil

cdef inline void drift_particle(particle_t *particle, real2_t dtds) nogil

cdef inline void rescale(real3_t *f, real_t qtmh) nogil
