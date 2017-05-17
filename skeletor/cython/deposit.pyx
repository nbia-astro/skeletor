from types cimport real_t, real2_t, real3_t, particle_t, grid_t
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free


def deposit_cic(particle_t[:] particles, real4_t[:,:] current,
            grid_t grid, real_t S):

    cdef int Np = particles.shape[0]
    cdef int ip

    cdef real2_t offset

    offset.x = grid.lbx - 0.5
    offset.y = grid.lby - 0.5 - grid.noff

    # Density deposition
    for ip in range(Np):
        deposit_particle_cic(particles[ip], current, grid, S, offset)

def deposit_tsc(particle_t[:] particles, real4_t[:,:] current,
            grid_t grid, real_t S):

    cdef int Np = particles.shape[0]
    cdef int ip

    cdef real2_t offset

    offset.x = grid.lbx - 0.5
    offset.y = grid.lby - 0.5 - grid.noff

    # Density deposition
    for ip in range(Np):
        deposit_particle_tsc(particles[ip], current, grid, S, offset)