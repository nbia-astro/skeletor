from ctypes cimport real_t, particle_t

def deposit (
        particle_t[:] particles, real_t[:,:] density,
        real_t charge, int noff, int nx, int ny, real_t St):

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    cdef Lx = <real_t> nx
    cdef Ly = <real_t> ny

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

        density[iy  , ix  ] += ty*tx
        density[iy  , ix+1] += ty*dx

        if iy != (ny-1):
            density[iy+1, ix  ] += dy*tx
            density[iy+1, ix+1] += dy*dx
        else:
            x += Ly*St
            while x >= Lx:
                x -= Lx
            while x < 0.0:
                x += Lx

            ix = <int> x
            dx = x - <real_t> ix
            tx = 1.0 - dx
            density[iy+1, ix  ] += dy*tx
            density[iy+1, ix+1] += dy*dx

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge
