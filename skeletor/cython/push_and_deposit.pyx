from types cimport real_t, real2_t, particle_t, grid_t
from particle_push cimport kick, gather_cic
from particle_push cimport drift2 as drift
from deposit cimport deposit_particle
from particle_boundary cimport periodic_x_cdef as periodic_x
from particle_boundary cimport calculate_ihole_cdef as calculate_ihole
from libc.math cimport fabs

def push_and_deposit(
         particle_t[:] particles, real2_t[:, :] E, real2_t[:, :] B,
         real_t qtmh, real_t dt, grid_t grid, int[:] ihole,
         real_t[:, :] density, real2_t[:,:] J, real_t S, const bint update):

    # Number of particles
    cdef int Np = particles.shape[0]

    # Electric and magnetic fields at particle location
    cdef real2_t e, b

    # A single particle struct
    cdef particle_t particle

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    # Variable needed by calculate_ihole
    cdef int ih = 0

    # TODO: Define this in types.pxd
    cdef real_t Lx = <real_t> grid.nx

    for ip in range(Np):
        # Copy particle data to temporary struct
        particle = particles[ip]

        # Gather and electric & magnetic fields with qtmh = 0.5*dt*charge/mass
        gather_cic(particle, E, &e, grid, qtmh)
        gather_cic(particle, B, &b, grid, qtmh)

        # Kick the particle velocities
        kick(&particle, e, b)

        # First half of particle drift
        drift(&particle, 0.5*dt, grid)

        # Make sure that particle has not moved more than a half grid cell
        if (fabs(particle.x - particles[ip].x) > 0.5) or \
           (fabs(particle.y - particles[ip].y) > 0.5):
            ihole[0] = -1

        # Deposit the particle
        deposit_particle(particle, density, J, grid, S)

        if update:
            # Second half of particle drift
            drift(&particle, 0.5*dt, grid)

            # Boundary conditions
            periodic_x(&particle, Lx)

            # Calculate ihole for use in PPIC2's ccpmove2
            ih = calculate_ihole(particle, ihole, grid, ih, ip)

            # Update value in the particle array
            particles[ip] = particle

    if update:
        # set end of file flag if it has not failed inside the main loop
        if ihole[0] >= 0:
            ihole[0] = ih
