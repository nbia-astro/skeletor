from ctypes cimport real_t, particle_t


def periodic_x(particle_t[:] particles, int nx):

    cdef real_t Lx = <real_t> nx

    for ip in range(particles.shape[0]):

        while particles[ip].x < 0.0:
            particles[ip].x += Lx
        while particles[ip].x >= Lx:
            particles[ip].x -= Lx


def periodic_y(particle_t[:] particles, int ny):

    cdef real_t Ly = <real_t> ny

    for ip in range(particles.shape[0]):

        while particles[ip].y < 0.0:
            particles[ip].y += Ly
        while particles[ip].y >= Ly:
            particles[ip].y -= Ly


def shear_periodic_y(particle_t[:] particles, int ny, real_t S, real_t t):
    """Shearing periodic boundaries along y.

       This function modifies x and vx and subsequently applies periodic
       boundaries on x.

       The periodic boundaries on y are handled by ppic2 *after* we have
       used the values of y to update x and vx.
    """
    cdef real_t Ly = <real_t> ny

    for ip in range(particles.shape[0]):
        # Left
        if particles[ip].y < 0.0:
            particles[ip].x -= S*Ly*t
            particles[ip].vx -= S*Ly
        # Right
        if particles[ip].y >= Ly:
            particles[ip].x += S*Ly*t
            particles[ip].vx += S*Ly
