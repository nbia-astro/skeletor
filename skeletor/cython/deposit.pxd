from types cimport real_t, real2_t, real4_t, particle_t, grid_t

cdef inline void deposit_particle(particle_t particle, real4_t[:,:] current,
                                  grid_t grid, real_t S, real2_t offset) nogil:

        cdef int ix, iy
        cdef real_t tx, ty, dx, dy
        cdef real_t x, y

        x = particle.x + offset.x
        y = particle.y + offset.y

        ix = <int> x
        iy = <int> y

        dx = x - <real_t> ix
        dy = y - <real_t> iy

        tx = 1.0 - dx
        ty = 1.0 - dy

        current[iy  , ix  ].t += ty*tx
        current[iy  , ix  ].x += ty*tx*(particle.vx + S*particle.y*grid.dy)
        current[iy  , ix  ].y += ty*tx*particle.vy
        current[iy  , ix  ].z += ty*tx*particle.vz

        current[iy  , ix+1].t += ty*dx
        current[iy  , ix+1].x += ty*dx*(particle.vx + S*particle.y*grid.dy)
        current[iy  , ix+1].y += ty*dx*particle.vy
        current[iy  , ix+1].z += ty*dx*particle.vz

        current[iy+1, ix  ].t += dy*tx
        current[iy+1, ix  ].x += dy*tx*(particle.vx + S*particle.y*grid.dy)
        current[iy+1, ix  ].y += dy*tx*particle.vy
        current[iy+1, ix  ].z += dy*tx*particle.vz

        current[iy+1, ix+1].t += dy*dx
        current[iy+1, ix+1].x += dy*dx*(particle.vx + S*particle.y*grid.dy)
        current[iy+1, ix+1].y += dy*dx*particle.vy
        current[iy+1, ix+1].z += dy*dx*particle.vz
