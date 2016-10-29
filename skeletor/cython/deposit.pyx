from ctypes cimport real_t, particle_t

def deposit (
        particle_t[:] particles, real_t[:,:] density,
        real_t charge, int noff, int lbx, int lby):

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    for ip in range(particles.shape[0]):

        x = particles[ip].x
        y = particles[ip].y

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

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge

def deposit_tsc(
        particle_t[:] particles, real_t[:,:] density,
        real_t charge, int noff, int lbx, int lby):

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    # cdef Index j
    # cdef Real r, wm, w0, wp

    # j = <Index> r
    # r -= <Real> j + 0.5
    # w0 = 0.75 - r*r
    # wp = 0.5*(0.5 + r)**2
    # wm = 1.0 - (w0 + wp)

    # sources[0,j - 1] += wm
    # sources[0,j    ] += w0
    # sources[0,j + 1] += wp

    for ip in range(particles.shape[0]):

        x = particles[ip].x
        y = particles[ip].y

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

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge