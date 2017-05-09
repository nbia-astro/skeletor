from types cimport real_t, real2_t, real3_t, particle_t, grid_t
from cython.parallel import prange

def boris_push(particle_t[:] particles, real3_t[:, :] E, real3_t[:, :] B,
               real_t qtmh, real_t dt, grid_t grid, const int order):

    cdef int Np = particles.shape[0]
    # Electric and magnetic fields at particle location
    cdef real3_t e, b

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    # CIC or TSC interpolation
    if order == 1:
        gather = gather_cic
    elif order == 2:
        gather = gather_tsc

    # Offset in interpolation for E and B-fields
    cdef real2_t offsetE, offsetB
    offsetB.x = grid.lbx
    offsetB.y = grid.lby - grid.noff
    offsetE.x = offsetB.x - 0.5
    offsetE.y = offsetB.y - 0.5

    # Because the particle position is stored in units of the grid spacing,
    # the drift velocity needs to be scaled by the grid spacing. This is
    # achieved easily by rescaling the time step.
    cdef real2_t dtds
    dtds.x = dt/grid.dx
    dtds.y = dt/grid.dy

    for ip in range(Np):
        # Gather and electric & magnetic fields
        gather(particles[ip], E, &e, offsetE)
        gather(particles[ip], B, &b, offsetB)

        # Rescale values with qtmh = 0.5*dt*charge/mass
        rescale(&e, qtmh)
        rescale(&b, qtmh)

        kick_particle(&particles[ip], e, b)
        drift_particle(&particles[ip], dtds)

def modified_boris_push(particle_t[:] particles, real3_t[:, :] E,
                        real3_t[:, :] B, real_t qtmh, real_t dt, grid_t grid,
                        real_t Omega, real_t S, const int order):

    cdef int Np = particles.shape[0]
    # Electric and magnetic fields at particle location
    cdef real3_t e, b

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    # CIC or TSC interpolation
    if order == 1:
        gather = gather_cic
    elif order == 2:
        gather = gather_tsc

    # Offset in interpolation for E and B-fields
    cdef real2_t offsetE, offsetB
    offsetB.x = grid.lbx
    offsetB.y = grid.lby - grid.noff
    offsetE.x = offsetB.x - 0.5
    offsetE.y = offsetB.y - 0.5

    # Because the particle position is stored in units of the grid spacing,
    # the drift velocity needs to be scaled by the grid spacing. This is
    # achieved easily by rescaling the time step.
    cdef real2_t dtds
    dtds.x = dt/grid.dx
    dtds.y = dt/grid.dy

    for ip in range(Np):
        # Gather and electric & magnetic fields
        gather(particles[ip], E, &e, offsetE)
        gather(particles[ip], B, &b, offsetB)

        # Rescale values with qtmh = 0.5*dt*charge/mass
        rescale(&e, qtmh)
        rescale(&b, qtmh)

        # Modify fields due to rotation and shear
        b.z = b.z + Omega*dt
        # TODO: We need to get this working/Make this more general
        e.y = e.y - S*particles[ip].y*grid.dy*b.z

        kick_particle(&particles[ip], e, b)
        drift_particle(&particles[ip], dtds)


def drift(particle_t[:] particles, real_t dt, grid_t grid):

    cdef int Np = particles.shape[0]
    cdef int ip
    cdef real2_t dtds

    dtds.x = dt/grid.dx
    dtds.y = dt/grid.dy

    for ip in prange(Np, nogil=True, schedule='static'):
        drift_particle(&particles[ip], dtds)
