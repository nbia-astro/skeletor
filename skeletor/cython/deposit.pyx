from types cimport real_t, real2_t, particle_t, grid_t
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free


def deposit(
        particle_t[:] particles, real_t[:, :] density, real2_t[:,:] J,
        grid_t grid, real_t S):

    cdef int Np = particles.shape[0]
    cdef int ip

    # Density deposition
    for ip in range(Np):
        deposit_particle(particles[ip], density, J, grid, S)

cdef inline void deposit_particle(particle_t particle, real_t[:,:] density,
                    real2_t[:,:] J, grid_t grid, real_t S) nogil:

        cdef int ix, iy
        cdef real_t tx, ty, dx, dy

        ix = <int> particle.x
        iy = <int> particle.y

        dx = particle.x - <real_t> ix
        dy = particle.y - <real_t> iy

        tx = 1.0 - dx
        ty = 1.0 - dy

        iy = iy - grid.noff

        ix = ix + grid.lbx
        iy = iy + grid.lby

        density[iy  , ix  ] += ty*tx
        density[iy  , ix+1] += ty*dx
        density[iy+1, ix  ] += dy*tx
        density[iy+1, ix+1] += dy*dx

        J[iy  , ix  ].x += ty*tx*(particle.vx + S*particle.y)
        J[iy  , ix+1].x += ty*dx*(particle.vx + S*particle.y)
        J[iy+1, ix  ].x += dy*tx*(particle.vx + S*particle.y)
        J[iy+1, ix+1].x += dy*dx*(particle.vx + S*particle.y)

        J[iy  , ix  ].y += ty*tx*particle.vy
        J[iy  , ix+1].y += ty*dx*particle.vy
        J[iy+1, ix  ].y += dy*tx*particle.vy
        J[iy+1, ix+1].y += dy*dx*particle.vy

        J[iy  , ix  ].z += ty*tx*particle.vz
        J[iy  , ix+1].z += ty*dx*particle.vz
        J[iy+1, ix  ].z += dy*tx*particle.vz
        J[iy+1, ix+1].z += dy*dx*particle.vz
