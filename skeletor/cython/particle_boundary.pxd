from types cimport real_t, particle_t, grid_t

cdef inline void periodic_x_cdef(particle_t *particle, real_t Lx) nogil:
    while particle.x < 0.0:
        particle.x = particle.x + Lx
    while particle.x >= Lx:
        particle.x = particle.x - Lx

cdef inline int calculate_ihole_cdef(particle_t particle, int[:] ihole,
                           grid_t grid, int ih, int ip) nogil:

    cdef int ntmax = ihole.shape[0] - 1

    if (particle.y < grid.edges[0]) or (particle.y >= grid.edges[1]):
        if ih < ntmax:
            ihole[ih+1] = ip + 1
        else:
            ihole[0] = -ih
        ih += 1

    return ih
