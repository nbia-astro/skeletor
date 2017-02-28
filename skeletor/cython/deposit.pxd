from types cimport real_t, real2_t, particle_t, grid_t

cdef inline void deposit_particle(particle_t particle, real_t[:,:] density,
                    real2_t[:,:] J, grid_t grid, real_t S) nogil