from types cimport real_t, real2_t, grid_t
cimport cython

def gradient(real_t[:,:] f, real2_t[:,:] grad, grid_t grid):

    cdef int ix, iy

    for iy in range(grid.lby, grid.uby):
        for ix in range(grid.lbx, grid.ubx):
            grad[iy, ix].x = 0.5*(f[iy, ix+1] - f[iy, ix-1])
            grad[iy, ix].y = 0.5*(f[iy+1, ix] - f[iy-1, ix])
            grad[iy, ix].z = 0.0

def curl_up(real_t[:,:] fx, real_t[:,:] fy, real_t[:,:] fz, real2_t[:,:] curl,
            grid_t grid):

    cdef int ix, iy

    for iy in range(grid.lby, grid.uby):
        for ix in range(grid.lbx, grid.ubx):
            curl[iy, ix].x =   ddyup(fz, ix, iy)
            curl[iy, ix].y = - ddxup(fz, ix, iy)
            curl[iy, ix].z = ddxup(fy, ix, iy) - ddyup(fx, ix, iy)

def curl_down(real_t[:,:] fx, real_t[:,:] fy, real_t[:,:] fz,
              real2_t[:,:] curl, grid_t grid):

    cdef int ix, iy

    for iy in range(grid.lby, grid.uby):
        for ix in range(grid.lbx, grid.ubx):
            curl[iy, ix].x =   ddydn(fz, ix, iy)
            curl[iy, ix].y = - ddxdn(fz, ix, iy)
            curl[iy, ix].z = ddxdn(fy, ix, iy) - ddydn(fx, ix, iy)

cdef inline real_t ddyup(real_t[:, :] f, int ix, int iy):
    return 0.5*(f[iy+1,ix+1] + f[iy+1,ix] - f[iy,ix+1] - f[iy,ix])

cdef inline real_t ddxup(real_t[:, :] f, int ix, int iy):
    return 0.5*(f[iy+1,ix+1] + f[iy,ix+1] - f[iy+1,ix] - f[iy,ix])

cdef inline real_t ddydn(real_t[:, :] f, int ix, int iy):
    return 0.5*(f[iy,ix] + f[iy,ix-1] - f[iy-1,ix] - f[iy-1,ix-1])

cdef inline real_t ddxdn(real_t[:, :] f, int ix, int iy):
    return 0.5*(f[iy,ix] + f[iy-1,ix] - f[iy,ix-1] - f[iy-1,ix-1])

def unstagger(real_t[:,:] fx, real_t[:,:] fy, real_t[:,:] fz, real2_t[:,:] g,
              grid_t grid):

    cdef int ix, iy

    for iy in range(grid.lby, grid.uby):
        for ix in range(grid.lbx, grid.ubx):
            g[iy, ix].x = inter_dn(fx, ix, iy)
            g[iy, ix].y = inter_dn(fy, ix, iy)
            g[iy, ix].z = inter_dn(fz, ix, iy)

def stagger(real_t[:,:] fx, real_t[:,:] fy, real_t[:,:] fz, real2_t[:,:] g,
              grid_t grid):

    cdef int ix, iy

    for iy in range(grid.lby, grid.uby):
        for ix in range(grid.lbx, grid.ubx):
            g[iy, ix].x = inter_up(fx, ix, iy)
            g[iy, ix].y = inter_up(fy, ix, iy)
            g[iy, ix].z = inter_up(fz, ix, iy)

cdef inline real_t inter_up(real_t[:, :] f, int ix, int iy):
    return 0.25*(f[iy+1,ix+1] + f[iy+1,ix] + f[iy,ix+1] + f[iy,ix])

cdef inline real_t inter_dn(real_t[:, :] f, int ix, int iy):
    return 0.25*(f[iy,ix] + f[iy-1,ix] + f[iy,ix-1] + f[iy-1,ix-1])


