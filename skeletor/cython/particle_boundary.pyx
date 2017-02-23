from types cimport real_t, particle_t, grid_t
from cython.parallel import prange


def periodic_x(particle_t[:] particles, int nx):
    cdef int Np = particles.shape[0]
    cdef real_t Lx = <real_t> nx
    cdef int ip

    for ip in prange(Np, nogil=True, schedule='static'):
        periodic_x_cdef(&particles[ip], Lx)

cdef inline void periodic_x_cdef(particle_t *particle, real_t Lx) nogil:
    while particle.x < 0.0:
        particle.x = particle.x + Lx
    while particle.x >= Lx:
        particle.x = particle.x - Lx

def calculate_ihole(particle_t[:] particles, int[:] ihole, grid_t grid):
    cdef int ih = 0
    cdef int nh = 0
    cdef int ntmax = ihole.shape[0] - 1
    cdef int ip

    for ip in range(particles.shape[0]):
        if (particles[ip].y < grid.edges[0]) or (particles[ip].y >= grid.edges[1]):
            if ih < ntmax:
                ihole[ih+1] = ip + 1
            else:
                nh = 1
            ih += 1
    # set end of file flag
    if nh > 0:
        ih = -ih
    ihole[0] = ih

def shear_periodic_y(particle_t[:] particles, int ny, real_t S, real_t t):
    """Shearing periodic boundaries along y.

       This function modifies x and vx and subsequently applies periodic
       boundaries on x.

       The periodic boundaries on y are handled by ppic2 *after* we have
       used the values of y to update x and vx.
    """
    cdef int Np = particles.shape[0]
    cdef real_t Ly = <real_t> ny
    cdef int ip

    for ip in prange(Np, nogil=True, schedule='static'):
        # Left
        if particles[ip].y < 0.0:
            particles[ip].x = particles[ip].x - S*Ly*t
            particles[ip].vx = particles[ip].vx - S*Ly
        # Right
        if particles[ip].y >= Ly:
            particles[ip].x = particles[ip].x + S*Ly*t
            particles[ip].vx = particles[ip].vx + S*Ly
