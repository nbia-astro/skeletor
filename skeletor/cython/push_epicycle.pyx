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


def push_cic(particle_t[:] particles, real2_t[:, :] E, real_t bz, real_t qtmh,
         real_t dt, int noff, int lbx, int lby):

    cdef real_t ex, ey

    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
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

        # Rescale electric & magnetic field with qtmh = 0.5*dt*charge/mass
        ex *= qtmh
        ey *= qtmh
        bz *= qtmh

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz + ex
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt

def push_tsc(particle_t[:] particles, real2_t[:, :] E, real_t bz, real_t qtmh,
         real_t dt, int noff, int lbx, int lby):

    cdef real_t ex, ey

    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
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

        # Rescale electric & magnetic field with qtmh = 0.5*dt*charge/mass
        ex *= qtmh
        ey *= qtmh
        bz *= qtmh

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz + ex
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt
