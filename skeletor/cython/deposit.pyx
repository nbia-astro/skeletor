from ctypes cimport real_t, real2_t, particle_t

def deposit(
        particle_t[:] particles, real_t[:,:] density, real2_t[:,:] J,
        real_t charge, int noff, int lbx, int lby, real_t S):

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty
    cdef real_t vx, vy

    for ip in range(particles.shape[0]):

        x = particles[ip].x
        y = particles[ip].y

        vx = particles[ip].vx + S*y
        vy = particles[ip].vy

        ix = <int> x
        iy = <int> y

        dx = x - <real_t> ix
        dy = y - <real_t> iy

        tx = 1.0 - dx
        ty = 1.0 - dy

        iy -= noff

        ix += lbx
        iy += lby

        density[iy  , ix  ] += ty*tx
        density[iy  , ix+1] += ty*dx
        density[iy+1, ix  ] += dy*tx
        density[iy+1, ix+1] += dy*dx

        J[iy  , ix  ].x += ty*tx*vx
        J[iy  , ix+1].x += ty*dx*vx
        J[iy+1, ix  ].x += dy*tx*vx
        J[iy+1, ix+1].x += dy*dx*vx

        J[iy  , ix  ].y += ty*tx*vy
        J[iy  , ix+1].y += ty*dx*vy
        J[iy+1, ix  ].y += dy*tx*vy
        J[iy+1, ix+1].y += dy*dx*vy

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge
