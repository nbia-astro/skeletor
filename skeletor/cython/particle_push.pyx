from types cimport real_t, real2_t, particle_t, grid_t
from cython.parallel import prange


def boris_push(particle_t[:] particles, real2_t[:, :] E, real_t bz,
               real_t qtmh, real_t dt, grid_t grid):

    cdef int Np = particles.shape[0]
    cdef real_t ex, ey
    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
    cdef real_t fac

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip, ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    # Rescale magnetic field with qtmh = 0.5*dt*charge/mass
    bz = qtmh*bz

    for ip in prange(Np, nogil=True, schedule='static'):

        # Interpolate field onto particle (TODO: Move to separate function)
        x = particles[ip].x
        y = particles[ip].y

        ix = <int> x
        iy = <int> y

        dx = x - <real_t> ix
        dy = y - <real_t> iy

        tx = 1.0 - dx
        ty = 1.0 - dy

        iy = iy - grid.noff

        ix = ix + grid.lbx
        iy = iy + grid.lby

        ex = dy*(dx*E[iy+1, ix+1].x + tx*E[iy+1, ix].x)  \
            + ty*(dx*E[iy, ix+1].x + tx*E[iy, ix].x)

        ey = dy*(dx*E[iy+1, ix+1].y + tx*E[iy+1, ix].y)  \
            + ty*(dx*E[iy, ix+1].y + tx*E[iy, ix].y)

        # Rescale electric field with qtmh = 0.5*dt*charge/mass
        ex = qtmh*ex
        ey = qtmh*ey

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz + ex
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x = particles[ip].x + particles[ip].vx*dt
        particles[ip].y = particles[ip].y + particles[ip].vy*dt


def modified_boris_push(particle_t[:] particles, real2_t[:, :] E, real_t bz,
                        real_t qtmh, real_t dt, grid_t grid,
                        real_t Omega, real_t S):

    cdef int Np = particles.shape[0]
    cdef real_t ex, ey
    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
    cdef real_t fac

    cdef int ip, ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    # Rescale magnetic field with qtmh = 0.5*dt*charge/mass
    bz = qtmh*bz

    # Modify fields due to rotation and shear
    bz = bz + Omega*dt

    for ip in prange(Np, nogil=True, schedule='static'):

        # Interpolate field onto particle (TODO: Move to separate function)
        x = particles[ip].x
        y = particles[ip].y

        ix = <int> x
        iy = <int> y

        dx = x - <real_t> ix
        dy = y - <real_t> iy

        tx = 1.0 - dx
        ty = 1.0 - dy

        iy = iy - grid.noff

        ix = ix + grid.lbx
        iy = iy + grid.lby

        ex = dy*(dx*E[iy+1, ix+1].x + tx*E[iy+1, ix].x)  \
            + ty*(dx*E[iy, ix+1].x + tx*E[iy, ix].x)

        ey = dy*(dx*E[iy+1, ix+1].y + tx*E[iy+1, ix].y)  \
            + ty*(dx*E[iy, ix+1].y + tx*E[iy, ix].y)

        # Rescale electric & magnetic field with qtmh = 0.5*dt*charge/mass
        ex = qtmh*ex
        ey = qtmh*ey

        # Modify fields due to rotation and shear
        ey = ey - S*y*bz

        vmx = particles[ip].vx + ex
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz + ex
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x = particles[ip].x + particles[ip].vx*dt
        particles[ip].y = particles[ip].y + particles[ip].vy*dt


def drift(particle_t[:] particles, real_t dt):

    cdef int Np = particles.shape[0]
    cdef int ip
    for ip in prange(Np, nogil=True, schedule='static'):

        particles[ip].x = particles[ip].x + particles[ip].vx*dt
        particles[ip].y = particles[ip].y + particles[ip].vy*dt
