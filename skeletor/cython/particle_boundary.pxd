from types cimport real_t, real2_t, particle_t, grid_t

cdef inline void periodic_x_cdef(particle_t *particle, real_t Lx) nogil

cdef inline int calculate_ihole_cdef(particle_t particle, int[:] ihole,
                           grid_t grid, int ih, int ip) nogil