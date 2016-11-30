from ctypes cimport real_t, real2_t, particle_t

def deposit_cic (
        particle_t[:] particles, real_t[:,:] density, real2_t[:,:] J,
        real_t charge, int noff, int lbx, int lby):

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty
    cdef real_t vx, vy, vz

    for ip in range(particles.shape[0]):

        x = particles[ip].x
        y = particles[ip].y

        vx = particles[ip].vx
        vy = particles[ip].vy
        vz = particles[ip].vz

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

        J[iy  , ix  ].z += ty*tx*vz
        J[iy  , ix+1].z += ty*dx*vz
        J[iy+1, ix  ].z += dy*tx*vz
        J[iy+1, ix+1].z += dy*dx*vz

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge

def deposit_tsc(
        particle_t[:] particles, real_t[:,:] density, real2_t[:,:] J,
        real_t charge, int noff, int lbx, int lby):

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t vx, vy, vz

    cdef real_t wmx, w0x, wpx, wmy, w0y, wpy

    for ip in range(particles.shape[0]):

        x = particles[ip].x
        y = particles[ip].y

        vx = particles[ip].vx
        vy = particles[ip].vy
        vz = particles[ip].vz

        x += 0.5
        y += 0.5

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

        J[iy-1 ,ix-1].x += wmy*wmx*vx
        J[iy-1 ,ix  ].x += wmy*w0x*vx
        J[iy-1 ,ix+1].x += wmy*wpx*vx
        J[iy  , ix-1].x += w0y*wmx*vx
        J[iy  , ix  ].x += w0y*w0x*vx
        J[iy  , ix+1].x += w0y*wpx*vx
        J[iy+1, ix-1].x += wpy*wmx*vx
        J[iy+1, ix  ].x += wpy*w0x*vx
        J[iy+1, ix+1].x += wpy*wpx*vx

        J[iy-1 ,ix-1].y += wmy*wmx*vy
        J[iy-1 ,ix  ].y += wmy*w0x*vy
        J[iy-1 ,ix+1].y += wmy*wpx*vy
        J[iy  , ix-1].y += w0y*wmx*vy
        J[iy  , ix  ].y += w0y*w0x*vy
        J[iy  , ix+1].y += w0y*wpx*vy
        J[iy+1, ix-1].y += wpy*wmx*vy
        J[iy+1, ix  ].y += wpy*w0x*vy
        J[iy+1, ix+1].y += wpy*wpx*vy

        J[iy-1 ,ix-1].z += wmy*wmx*vz
        J[iy-1 ,ix  ].z += wmy*w0x*vz
        J[iy-1 ,ix+1].z += wmy*wpx*vz
        J[iy  , ix-1].z += w0y*wmx*vz
        J[iy  , ix  ].z += w0y*w0x*vz
        J[iy  , ix+1].z += w0y*wpx*vz
        J[iy+1, ix-1].z += wpy*wmx*vz
        J[iy+1, ix  ].z += wpy*w0x*vz
        J[iy+1, ix+1].z += wpy*wpx*vz

    for iy in range(density.shape[0]):
        for ix in range(density.shape[1]):
            density[iy, ix] *= charge