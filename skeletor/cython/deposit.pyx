from types cimport real_t, real2_t, particle_t, grid_t
from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport abort, malloc, free

from types import Float
from libc.stdio cimport printf
from numpy import empty
cimport openmp
cimport cython
from numpy import sum as np_sum

cdef class CythonDeposit:

    cdef int num_threads
    cdef grid_t grid
    cdef real_t[:,:,:,:] current

    def __cinit__(self, grid_t grid):
        # Determine how many OpenMP threads are running
        with nogil, parallel():
            self.num_threads = openmp.omp_get_num_threads()
        # Store grid
        self.grid = grid
        # The dimensions of the current array are such that the "four-vector"
        # components vary the fastest
        shape = self.num_threads, grid.nypmx, grid.nxpmx, 3
        self.current = empty(shape, dtype=Float)

    @cython.initializedcheck(False)
    def __call__(self, particle_t[:] particles,
            real_t[:,:] rho, real2_t[:,:] J, real_t charge, real_t S):

        cdef int N = particles.shape[0]

        cdef int thid, n
        cdef real_t x, y, vx, vy
        cdef int ix, iy
        cdef real_t dx, dy, tx, ty

        with nogil, parallel():

            # OpenMP thread ID
            # (I hope this is always an integer
            # running from 0 to num_threads-1)
            thid = threadid()

            # OpenMP-parallelize over the particle index
            for n in prange(N, schedule='static'):

                # Particle position in units of the grid spacing
                x = particles[n].x
                y = particles[n].y

                # Particle velocity (relative to the background shear)
                vx = particles[n].vx + S*y
                vy = particles[n].vy

                # Integer part of particle position
                ix = <int> x
                iy = <int> y

                # Fractional part/weight
                dx = x - <real_t> ix
                dy = y - <real_t> iy

                # Complimentary weight
                tx = 1.0 - dx
                ty = 1.0 - dy

                # Account for guard cells and domain decomposition when
                # indexing the charge and current array
                ix = ix + self.grid.lbx
                iy = iy + self.grid.lby - self.grid.noff

                # Deposit charge and current. Note that each OpenMP thread
                # deposits into its own section of the "global" charge and
                # current array
                self.current[thid, iy  , ix  , 0] += ty*tx
                self.current[thid, iy  , ix  , 1] += ty*tx*vx
                self.current[thid, iy  , ix  , 2] += ty*tx*vy

                self.current[thid, iy  , ix+1, 0] += ty*dx
                self.current[thid, iy  , ix+1, 1] += ty*dx*vx
                self.current[thid, iy  , ix+1, 2] += ty*dx*vy

                self.current[thid, iy+1, ix  , 0] += dy*tx
                self.current[thid, iy+1, ix  , 1] += dy*tx*vx
                self.current[thid, iy+1, ix  , 2] += dy*tx*vy

                self.current[thid, iy+1, ix+1, 0] += dy*dx
                self.current[thid, iy+1, ix+1, 1] += dy*dx*vx
                self.current[thid, iy+1, ix+1, 2] += dy*dx*vy

        with nogil, parallel():

            # Accumulate deposited charge and current densities from all OpenMP
            # processes. OpenMP-parallelize over the y-direction this time
            #
            # First (i.e. zeroth) OpenMP thread
            thid = 0
            for iy in prange(self.grid.nypmx, schedule='static'):
                for ix in range(self.grid.nxpmx):
                    rho[iy, ix] = self.current[thid, iy, ix, 0]
                    J[iy, ix].x = self.current[thid, iy, ix, 1]
                    J[iy, ix].y = self.current[thid, iy, ix, 2]
            # Add contributions from remaining threads
            for thid in range(1, self.num_threads):
                for iy in prange(self.grid.nypmx, schedule='static'):
                    for ix in range(self.grid.nxpmx):
                        rho[iy, ix] += self.current[thid, iy, ix, 0]
                        J[iy, ix].x += self.current[thid, iy, ix, 1]
                        J[iy, ix].y += self.current[thid, iy, ix, 2]

        # TODO: Are these separate "nogil, parallel" blocks necessary?
        with nogil, parallel():

            # Rescale charge density by the particle charge itself
            for iy in prange(self.grid.nypmx, schedule='static'):
                for ix in range(self.grid.nxpmx):
                    rho[iy, ix] *= charge
            # Do the same with the current density
            for iy in prange(self.grid.nypmx, schedule='static'):
                for ix in range(self.grid.nxpmx):
                    J[iy, ix].x *= charge
                    J[iy, ix].y *= charge


def deposit(
        particle_t[:] particles, real_t[:,:] density, real2_t[:,:] J,
        real_t charge, grid_t grid, real_t S):

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

            iy = iy - grid.noff

            ix = ix + grid.lbx
            iy = iy + grid.lby

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
