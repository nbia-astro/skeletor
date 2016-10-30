from ctypes cimport real_t, particle_t

def deposit_cic (
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

    cdef real_t wmx, w0x, wpx, wmy, w0y, wpy

    for ip in range(particles.shape[0]):

        x = particles[ip].x
        y = particles[ip].y

        ix = <int> x
        iy = <int> y

        dx = x - <real_t> ix
        dy = y - <real_t> iy

        w0x = 0.75 - dx*dx
        wpx = 0.5*(0.5 + dx)**2
        wmx = 1.0 - (w0x + wpx)

        w0y = 0.75 - dy*dy
        wpy = 0.5*(0.5 + dy)**2
        wmy = 1.0 - (w0y + wpy)

        iy -= noff

        ix += lbx
        iy += lby

        density[iy-1 ,ix-1] += wmy*wmx
        density[iy-1 ,ix  ] += wmy*w0x
        density[iy-1 ,ix+1] += wmy*wpx
        density[iy  , ix-1] += w0y*wmx
        density[iy  , ix  ] += w0y*w0x
        density[iy  , ix+1] += w0y*wpx
        density[iy+1, ix-1] += wpy*wmx
        density[iy+1, ix  ] += wpy*w0x
        density[iy+1, ix+1] += wpy*wpx

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge