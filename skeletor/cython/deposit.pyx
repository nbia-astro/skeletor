from types cimport real_t, real2_t, real3_t, particle_t, grid_t
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free


def deposit(particle_t[:] particles, real4_t[:,:] current,
            grid_t grid, real_t S, const int order):

    cdef int Np = particles.shape[0]
    cdef int ip

    cdef real2_t offset

    # CIC or TSC interpolation
    if order == 1:
        deposit_particle = deposit_particle_cic
    elif order == 2:
        deposit_particle = deposit_particle_tsc

    offset.x = grid.lbx - 0.5
    offset.y = grid.lby - 0.5 - grid.noff

    # Density deposition
    for ip in range(Np):
        deposit_particle(particles[ip], current, grid, S, offset)
