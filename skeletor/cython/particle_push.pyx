from ctypes cimport real_t, real2_t, particle_t


def boris_push(particle_t[:] particles, real2_t[:, :] E, real_t bz,
               real_t qtmh, real_t dt, int noff, int lbx, int lby):

    cdef real_t ex, ey

    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
    cdef real_t fac

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    # Rescale magnetic field with qtmh = 0.5*dt*charge/mass
    bz *= qtmh

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

        # Rescale electric field with qtmh = 0.5*dt*charge/mass
        ex *= qtmh
        ey *= qtmh

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz + ex
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt


def modified_boris_push(particle_t[:] particles, real2_t[:, :] E, real_t bz,
                        real_t qtmh, real_t dt, int noff, int lbx, int lby,
                        real_t Omega, real_t S):

    cdef real_t ex, ey

    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
    cdef real_t fac

    cdef int ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    # Rescale magnetic field with qtmh = 0.5*dt*charge/mass
    bz *= qtmh

    # Modify fields due to rotation and shear
    bz += Omega*dt

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

        # Modify fields due to rotation and shear
        ey -= S*y*bz

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz + ex
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt


def drift(particle_t[:] particles, real_t dt):

    for ip in range(particles.shape[0]):

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt
