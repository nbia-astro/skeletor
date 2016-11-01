from ctypes cimport real_t, real2_t

def gradient(real_t[:,:] f, real2_t[:,:] grad, int lbx, int ubx, int lby,
             int uby):

    cdef int ix, iy

    for iy in range(lby, uby):
        for ix in range(lbx, ubx):
            grad[iy, ix].x = 0.5*(f[iy, ix+1] - f[iy, ix-1])
            grad[iy, ix].y = 0.5*(f[iy+1, ix] - f[iy-1, ix])
