from types cimport real_t, real2_t, grid_t

def gradient(real_t[:,:] f, real2_t[:,:] grad, grid_t grid):

    cdef int ix, iy

    for iy in range(grid.lby, grid.uby):
        for ix in range(grid.lbx, grid.ubx):
            grad[iy, ix].x = 0.5*(f[iy, ix+1] - f[iy, ix-1])
            grad[iy, ix].y = 0.5*(f[iy+1, ix] - f[iy-1, ix])
