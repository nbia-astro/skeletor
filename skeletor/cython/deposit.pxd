from types cimport real_t, real2_t, real4_t, particle_t, grid_t

cdef inline void deposit_particle_cic(particle_t particle, real4_t[:,:]
                 current, grid_t grid, real_t S, real2_t offset) nogil:

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

cdef inline void deposit_particle_tsc(particle_t particle, real4_t[:,:]
                 current, grid_t grid, real_t S, real2_t offset) nogil:

        cdef int ix, iy
        cdef real_t dx, dy
        cdef real_t x, y
        cdef real_t wmx, w0x, wpx, wmy, w0y, wpy

        x = particle.x + offset.x
        y = particle.y + offset.y

        x = x + 0.5
        y = y + 0.5

        ix = <int> x
        iy = <int> y

        dx = x - <real_t> ix - 0.5
        dy = y - <real_t> iy - 0.5

        w0x = 0.75 - dx*dx
        wpx = 0.5*(0.5 + dx)**2
        wmx = 1.0 - (w0x + wpx)

        w0y = 0.75 - dy*dy
        wpy = 0.5*(0.5 + dy)**2
        wmy = 1.0 - (w0y + wpy)

        # TODO: Order these for efficient memory access

        current[iy-1 ,ix-1].t += wmy*wmx
        current[iy-1 ,ix  ].t += wmy*w0x
        current[iy-1 ,ix+1].t += wmy*wpx
        current[iy  , ix-1].t += w0y*wmx
        current[iy  , ix  ].t += w0y*w0x
        current[iy  , ix+1].t += w0y*wpx
        current[iy+1, ix-1].t += wpy*wmx
        current[iy+1, ix  ].t += wpy*w0x
        current[iy+1, ix+1].t += wpy*wpx

        current[iy-1 ,ix-1].x += wmy*wmx*(particle.vx + S*particle.y*grid.dy)
        current[iy-1 ,ix  ].x += wmy*w0x*(particle.vx + S*particle.y*grid.dy)
        current[iy-1 ,ix+1].x += wmy*wpx*(particle.vx + S*particle.y*grid.dy)
        current[iy  , ix-1].x += w0y*wmx*(particle.vx + S*particle.y*grid.dy)
        current[iy  , ix  ].x += w0y*w0x*(particle.vx + S*particle.y*grid.dy)
        current[iy  , ix+1].x += w0y*wpx*(particle.vx + S*particle.y*grid.dy)
        current[iy+1, ix-1].x += wpy*wmx*(particle.vx + S*particle.y*grid.dy)
        current[iy+1, ix  ].x += wpy*w0x*(particle.vx + S*particle.y*grid.dy)
        current[iy+1, ix+1].x += wpy*wpx*(particle.vx + S*particle.y*grid.dy)

        current[iy-1 ,ix-1].y += wmy*wmx*particle.vy
        current[iy-1 ,ix  ].y += wmy*w0x*particle.vy
        current[iy-1 ,ix+1].y += wmy*wpx*particle.vy
        current[iy  , ix-1].y += w0y*wmx*particle.vy
        current[iy  , ix  ].y += w0y*w0x*particle.vy
        current[iy  , ix+1].y += w0y*wpx*particle.vy
        current[iy+1, ix-1].y += wpy*wmx*particle.vy
        current[iy+1, ix  ].y += wpy*w0x*particle.vy
        current[iy+1, ix+1].y += wpy*wpx*particle.vy

        current[iy-1 ,ix-1].z += wmy*wmx*particle.vz
        current[iy-1 ,ix  ].z += wmy*w0x*particle.vz
        current[iy-1 ,ix+1].z += wmy*wpx*particle.vz
        current[iy  , ix-1].z += w0y*wmx*particle.vz
        current[iy  , ix  ].z += w0y*w0x*particle.vz
        current[iy  , ix+1].z += w0y*wpx*particle.vz
        current[iy+1, ix-1].z += wpy*wmx*particle.vz
        current[iy+1, ix  ].z += wpy*w0x*particle.vz
        current[iy+1, ix+1].z += wpy*wpx*particle.vz

