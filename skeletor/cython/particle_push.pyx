from types cimport real_t, real2_t, particle_t, grid_t
from cython.parallel import prange

def boris_push(particle_t[:] particles, real2_t[:, :] E, real2_t[:, :] B,
               real_t qtmh, real_t dt, grid_t grid):

    cdef int Np = particles.shape[0]
    cdef real_t ex, ey, ez
    cdef real_t bx, by, bz

    cdef real_t vmx, vmy, vmz
    cdef real_t vpx, vpy, vpz
    cdef real_t fac

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip, ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    for ip in range(Np):

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

        ez = dy*(dx*E[iy+1, ix+1].z + tx*E[iy+1, ix].z)  \
            + ty*(dx*E[iy, ix+1].z + tx*E[iy, ix].z)

        bx = dy*(dx*B[iy+1, ix+1].x + tx*B[iy+1, ix].x)  \
            + ty*(dx*B[iy, ix+1].x + tx*B[iy, ix].x)

        by = dy*(dx*B[iy+1, ix+1].y + tx*B[iy+1, ix].y)  \
            + ty*(dx*B[iy, ix+1].y + tx*B[iy, ix].y)

        bz = dy*(dx*B[iy+1, ix+1].z + tx*B[iy+1, ix].z)  \
            + ty*(dx*B[iy, ix+1].z + tx*B[iy, ix].z)

        # Rescale electric & magnetic field with qtmh = 0.5*dt*charge/mass
        ex = ex*qtmh
        ey = ey*qtmh
        ez = ez*qtmh
        bx = bx*qtmh
        by = by*qtmh
        bz = bz*qtmh

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

        particles[ip].x = particles[ip].x + particles[ip].vx*dt
        particles[ip].y = particles[ip].y + particles[ip].vy*dt


def modified_boris_push(particle_t[:] particles, real2_t[:, :] E,
                        real2_t[:, :] B, real_t qtmh, real_t dt, grid_t grid,
                        real_t Omega, real_t S):

    cdef int Np = particles.shape[0]
    cdef real_t ex, ey, ez
    cdef real_t bx, by, bz

    cdef real_t vmx, vmy, vmz
    cdef real_t vpx, vpy, vpz
    cdef real_t fac

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip, ix, iy
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty

    for ip in range(Np):

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

        ez = dy*(dx*E[iy+1, ix+1].z + tx*E[iy+1, ix].z)  \
            + ty*(dx*E[iy, ix+1].z + tx*E[iy, ix].z)

        bx = dy*(dx*B[iy+1, ix+1].x + tx*B[iy+1, ix].x)  \
            + ty*(dx*B[iy, ix+1].x + tx*B[iy, ix].x)

        by = dy*(dx*B[iy+1, ix+1].y + tx*B[iy+1, ix].y)  \
            + ty*(dx*B[iy, ix+1].y + tx*B[iy, ix].y)

        bz = dy*(dx*B[iy+1, ix+1].z + tx*B[iy+1, ix].z)  \
            + ty*(dx*B[iy, ix+1].z + tx*B[iy, ix].z)

        # Rescale electric & magnetic field with qtmh = 0.5*dt*charge/mass
        ex = ex*qtmh
        ey = ey*qtmh
        ez = ez*qtmh
        bx = bx*qtmh
        by = by*qtmh
        bz = bz*qtmh

        # Modify fields due to rotation and shear
        bz = bz + Omega*dt
        ey = ey - S*y*bz

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

        particles[ip].x = particles[ip].x + particles[ip].vx*dt
        particles[ip].y = particles[ip].y + particles[ip].vy*dt


def drift(particle_t[:] particles, real_t dt):

    cdef int Np = particles.shape[0]
    cdef int ip
    for ip in prange(Np, nogil=True, schedule='static'):

        particles[ip].x = particles[ip].x + particles[ip].vx*dt
        particles[ip].y = particles[ip].y + particles[ip].vy*dt
