from types cimport real_t, particle_t, grid_t
from cython.parallel import prange


def periodic_x(particle_t[:] particles, grid_t grid):
    cdef int Np = particles.shape[0]
    cdef real_t nx = <real_t> grid.nx  # Avoid mixed-type arithmetic
    cdef int ip

    for ip in prange(Np, nogil=True, schedule='static'):
        periodic_x_cdef(&particles[ip], nx)


def calculate_ihole(particle_t[:] particles, int[:] ihole, grid_t grid):
    cdef int ih = 0
    cdef int ip

    for ip in range(particles.shape[0]):
        ih = calculate_ihole_cdef(particles[ip], ihole, grid, ih, ip)

    # set end of file flag if it has not failed inside the loop
    if ihole[0] >= 0:
        ihole[0] = ih


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
