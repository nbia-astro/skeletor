from types cimport real_t, real2_t, real3_t, particle_t, grid_t

cdef inline void deposit_particle(particle_t particle, real_t[:,:] density,
                                  real3_t[:,:] J, grid_t grid, real_t S,
                                  real2_t offset) nogil:

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

        density[iy  , ix  ] += ty*tx
        density[iy  , ix+1] += ty*dx
        density[iy+1, ix  ] += dy*tx
        density[iy+1, ix+1] += dy*dx

        # TODO: We need to get this working/Make this more general
        J[iy  , ix  ].x += ty*tx*(particle.vx + S*particle.y*grid.dy)
        J[iy  , ix+1].x += ty*dx*(particle.vx + S*particle.y*grid.dy)
        J[iy+1, ix  ].x += dy*tx*(particle.vx + S*particle.y*grid.dy)
        J[iy+1, ix+1].x += dy*dx*(particle.vx + S*particle.y*grid.dy)

        J[iy  , ix  ].y += ty*tx*particle.vy
        J[iy  , ix+1].y += ty*dx*particle.vy
        J[iy+1, ix  ].y += dy*tx*particle.vy
        J[iy+1, ix+1].y += dy*dx*particle.vy

        J[iy  , ix  ].z += ty*tx*particle.vz
        J[iy  , ix+1].z += ty*dx*particle.vz
        J[iy+1, ix  ].z += dy*tx*particle.vz
        J[iy+1, ix+1].z += dy*dx*particle.vz
