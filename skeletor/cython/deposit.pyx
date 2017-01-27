from ctypes cimport real_t, real2_t, particle_t
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free

def deposit(
        particle_t[:] particles, real_t[:,:] density, real2_t[:,:] J,
        real_t charge, int noff, int lbx, int lby, real_t S):

    cdef int Np = particles.shape[0]
    cdef int ip, ix, iy, k, ky, kx
    cdef real_t x, y
    cdef real_t dx, dy
    cdef real_t tx, ty
    cdef real_t vx, vy
    cdef int nyp = density.shape[0]
    cdef int nxp = density.shape[1]
    cdef int size = nxp*nyp

    with nogil, parallel():
        # Deposition to local 1D arrays
        den_local = <real_t *>malloc(size*sizeof(real_t))
        Jx_local = <real_t *>malloc(size*sizeof(real_t))
        Jy_local = <real_t *>malloc(size*sizeof(real_t))

        for k in range(size):
                den_local[k] = 0.0
                Jx_local[k] = 0.0
                Jy_local[k] = 0.0

        for ip in prange(Np, schedule='static'):

            x = particles[ip].x
            y = particles[ip].y

            # Calculate the fluctuating x-velocity
            vx = particles[ip].vx + S*y
            vy = particles[ip].vy

            ix = <int> x
            iy = <int> y

            dx = x - <real_t> ix
            dy = y - <real_t> iy

            tx = 1.0 - dx
            ty = 1.0 - dy

            iy = iy - noff

            ix = ix + lbx
            iy = iy + lby

            # Store values in 1D arrays.
            # We use that the position [iy, ix] in the matrix corresponds to
            # the position [iy*nxp + ix] in the 1D arrays.
            den_local[iy*nxp + ix] += ty*tx
            den_local[iy*nxp + ix+1] += ty*dx
            den_local[(iy+1)*nxp + ix] += dy*tx
            den_local[(iy+1)*nxp + ix+1] += dy*dx

            Jx_local[iy*nxp + ix] += ty*tx*vx
            Jx_local[iy*nxp + ix+1] += ty*dx*vx
            Jx_local[(iy+1)*nxp + ix] += dy*tx*vx
            Jx_local[(iy+1)*nxp + ix+1] += dy*dx*vx

            Jy_local[iy*nxp + ix] += ty*tx*vy
            Jy_local[iy*nxp + ix+1] += ty*dx*vy
            Jy_local[(iy+1)*nxp + ix] += dy*tx*vy
            Jy_local[(iy+1)*nxp + ix+1] += dy*dx*vy

        # Add up contributions from each processor
        with gil:
            for ky in range(nyp):
                for kx in range(nxp):
                    density[ky, kx] += den_local[ky*nxp + kx]*charge
                    J[ky, kx].x += Jx_local[ky*nxp + kx]
                    J[ky, kx].y += Jy_local[ky*nxp + kx]
            # Free memory
            free(den_local)
            free(Jx_local)
            free(Jy_local)
