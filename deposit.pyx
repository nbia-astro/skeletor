from ctypes cimport particle_t
from cython cimport floating

def deposit (
        particle_t[:] particles, floating[:,:] density,
        floating charge, int noff):

    cdef int ix, iy
    cdef floating x, y
    cdef floating dx, dy
    cdef floating tx, ty

    for ip in range(particles.shape[0]):

        x = particles[ip].x
        y = particles[ip].y

        ix = <int> x
        iy = <int> y

        dx = x - <floating> ix
        dy = y - <floating> iy

        tx = 1.0 - dx
        ty = 1.0 - dy

        iy -= noff

        density[iy  , ix  ] += ty*tx
        density[iy  , ix+1] += ty*dx
        density[iy+1, ix  ] += dy*tx
        density[iy+1, ix+1] += dy*dx

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge
