from ctypes cimport real_t, real2_t, particle_t

def push_epicycle(particle_t[:] particles, real_t dt):

    cdef real_t ey, bz

    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
    cdef real_t fac

    cdef real_t Omega = 1.0
    cdef real_t S = -1.5

    for ip in range(particles.shape[0]):

        bz = Omega*dt
        ey = -S*particles[ip].y*bz

        vmx = particles[ip].vx
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt


def push_cic(particle_t[:] particles, real2_t[:, :] E, real2_t[:, :] B, real_t qtmh,
         real_t dt, int noff, int lbx, int lby):

    cdef real_t ex, ey, ez
    cdef real_t bx, by, bz

    cdef real_t vmx, vmy, vmz
    cdef real_t vpx, vpy, vpz
    cdef real_t fac

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    for ip in range(particles.shape[0]):

        # Interpolate field onto particle (TODO: Move to separate function)
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

        ex = dy*(dx*E[iy+1, ix+1].x + tx*E[iy+1, ix].x)  \
            + ty*(dx*E[iy, ix+1].x + tx*E[iy, ix].x)

        ey = dy*(dx*E[iy+1, ix+1].y + tx*E[iy+1, ix].y)  \
            + ty*(dx*E[iy, ix+1].y + tx*E[iy, ix].y)

        ez = dy*(dx*E[iy+1, ix+1].z + tx*E[iy+1, ix].z)  \
            + ty*(dx*E[iy, ix+1].z + tx*E[iy, ix].z)

        bx = dy*(dx*B[iy+1, ix+1].x + tx*B[iy+1, ix].x)  \
            + ty*(dx*B[iy, ix+1].x + tx*B[iy, ix].x)

        by = dy*(dx*B[iy+1, ix+1].y + tx*B[iy+1, ix].y)  \
            + ty*(dx*B[iy, ix+1].y + tx*B[iy, ix].y)

        bz = dy*(dx*B[iy+1, ix+1].z + tx*B[iy+1, ix].z)  \
            + ty*(dx*B[iy, ix+1].z + tx*B[iy, ix].z)

        # Rescale electric & magnetic field with qtmh = 0.5*dt*charge/mass
        ex *= qtmh
        ey *= qtmh
        ez *= qtmh
        bx *= qtmh
        by *= qtmh
        bz *= qtmh

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey
        vmz = particles[ip].vz + ez

        vpx = vmx + (vmy*bz - vmz*by)
        vpy = vmy + (vmz*bx - vmx*bz)
        vpz = vmz + (vmx*by - vmy*bx)

        fac = 2./(1. + bx*bx + by*by + bz*bz)

        particles[ip].vx = vmx + fac*(vpy*bz - vpz*by) + ex
        particles[ip].vy = vmy + fac*(vpz*bx - vpx*bz) + ey
        particles[ip].vz = vmz + fac*(vpx*by - vpy*bx) + ez

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt

def push_tsc(particle_t[:] particles, real2_t[:, :] E, real2_t[:,:] B, real_t qtmh,
         real_t dt, int noff, int lbx, int lby):

    cdef real_t ex, ey, ez
    cdef real_t bx, by, bz

    cdef real_t vmx, vmy, vmz
    cdef real_t vpx, vpy, vpz
    cdef real_t fac

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t wmx, w0x, wpx, wmy, w0y, wpy

    for ip in range(particles.shape[0]):

        # Interpolate field onto particle (TODO: Move to separate function)
        x = particles[ip].x
        y = particles[ip].y

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

        ex = wmy*(wmx*E[iy-1, ix-1].x+w0x*E[iy-1, ix  ].x+wpx*E[iy-1, ix+1].x)\
           + w0y*(wmx*E[iy  , ix-1].x+w0x*E[iy  , ix  ].x+wpx*E[iy  , ix+1].x)\
           + wpy*(wmx*E[iy+1, ix-1].x+w0x*E[iy+1, ix  ].x+wpx*E[iy+1, ix+1].x)

        ey = wmy*(wmx*E[iy-1, ix-1].y+w0x*E[iy-1, ix  ].y+wpx*E[iy-1, ix+1].y)\
           + w0y*(wmx*E[iy  , ix-1].y+w0x*E[iy  , ix  ].y+wpx*E[iy  , ix+1].y)\
           + wpy*(wmx*E[iy+1, ix-1].y+w0x*E[iy+1, ix  ].y+wpx*E[iy+1, ix+1].y)

        ez = wmy*(wmx*E[iy-1, ix-1].z+w0x*E[iy-1, ix  ].z+wpx*E[iy-1, ix+1].z)\
           + w0y*(wmx*E[iy  , ix-1].z+w0x*E[iy  , ix  ].z+wpx*E[iy  , ix+1].z)\
           + wpy*(wmx*E[iy+1, ix-1].z+w0x*E[iy+1, ix  ].z+wpx*E[iy+1, ix+1].z)

        bx = wmy*(wmx*B[iy-1, ix-1].x+w0x*B[iy-1, ix  ].x+wpx*B[iy-1, ix+1].x)\
           + w0y*(wmx*B[iy  , ix-1].x+w0x*B[iy  , ix  ].x+wpx*B[iy  , ix+1].x)\
           + wpy*(wmx*B[iy+1, ix-1].x+w0x*B[iy+1, ix  ].x+wpx*B[iy+1, ix+1].x)

        by = wmy*(wmx*B[iy-1, ix-1].y+w0x*B[iy-1, ix  ].y+wpx*B[iy-1, ix+1].y)\
           + w0y*(wmx*B[iy  , ix-1].y+w0x*B[iy  , ix  ].y+wpx*B[iy  , ix+1].y)\
           + wpy*(wmx*B[iy+1, ix-1].y+w0x*B[iy+1, ix  ].y+wpx*B[iy+1, ix+1].y)

        bz = wmy*(wmx*B[iy-1, ix-1].z+w0x*B[iy-1, ix  ].z+wpx*B[iy-1, ix+1].z)\
           + w0y*(wmx*B[iy  , ix-1].z+w0x*B[iy  , ix  ].z+wpx*B[iy  , ix+1].z)\
           + wpy*(wmx*B[iy+1, ix-1].z+w0x*B[iy+1, ix  ].z+wpx*B[iy+1, ix+1].z)

        # Rescale electric & magnetic field with qtmh = 0.5*dt*charge/mass
        ex *= qtmh
        ey *= qtmh
        ez *= qtmh
        bx *= qtmh
        by *= qtmh
        bz *= qtmh

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey
        vmz = particles[ip].vz + ez

        vpx = vmx + (vmy*bz - vmz*by)
        vpy = vmy + (vmz*bx - vmx*bz)
        vpz = vmz + (vmx*by - vmy*bx)

        fac = 2./(1. + bx*bx + by*by + bz*bz)

        particles[ip].vx = vmx + fac*(vpy*bz - vpz*by) + ex
        particles[ip].vy = vmy + fac*(vpz*bx - vpx*bz) + ey
        particles[ip].vz = vmz + fac*(vpx*by - vpy*bx) + ez

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt
