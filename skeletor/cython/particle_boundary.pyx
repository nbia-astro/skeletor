from types cimport real_t, particle_t, grid_t
from cython.parallel import prange


def periodic_x(particle_t[:] particles, grid_t grid):
    cdef int Np = particles.shape[0]
    cdef real_t nx = <real_t> grid.nx  # Avoid mixed-type arithmetic
    cdef int ip

    for ip in prange(Np, nogil=True, schedule='static'):
        periodic_x_cdef(&particles[ip], nx)

cdef inline void periodic_x_cdef(particle_t *particle, real_t nx) nogil:
    while particle.x < 0.0:
        particle.x = particle.x + nx
    while particle.x >= nx:
        particle.x = particle.x - nx

def calculate_ihole(particle_t[:] particles, int[:] ihole, grid_t grid):
    cdef int ih = 0
    cdef int ip

    for ip in range(particles.shape[0]):
        ih = calculate_ihole_cdef(particles[ip], ihole, grid, ih, ip)

    # set end of file flag if it has not failed inside the loop
    if ihole[0] >= 0:
        ihole[0] = ih

cdef inline int calculate_ihole_cdef(particle_t particle, int[:] ihole,
                           grid_t grid, int ih, int ip) nogil:

    cdef int ntmax = ihole.shape[0] - 1

    if (particle.y < grid.edges[0]) or (particle.y >= grid.edges[1]):
        if ih < ntmax:
            ihole[ih+1] = ip + 1
        else:
            ihole[0] = -ih
        ih += 1

    return ih


def shear_periodic_y(particle_t[:] particles, grid_t grid, real_t S, real_t t):
    """Shearing periodic boundaries along y.

       This function modifies x and vx and subsequently applies periodic
       boundaries on x.

       The periodic boundaries on y are handled by ppic2 *after* we have
       used the values of y to update x and vx.
    """
    cdef int Np = particles.shape[0]
    cdef real_t ny = <real_t> grid.ny
    cdef real_t vx_boost = S*grid.Ly
    cdef real_t x_boost = vx_boost*t/grid.dx
    cdef int ip

    for ip in prange(Np, nogil=True, schedule='static'):
        # Left
        if particles[ip].y < 0.0:
            particles[ip].x = particles[ip].x - x_boost
            particles[ip].vx = particles[ip].vx - vx_boost
        # Right
        if particles[ip].y >= ny:
            particles[ip].x = particles[ip].x + x_boost
            particles[ip].vx = particles[ip].vx + vx_boost
