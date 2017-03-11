from types cimport real_t, real2_t, real3_t, real4_t, particle_t, grid_t
from particle_push cimport gather_cic, rescale
from particle_push cimport drift_particle as drift
from particle_push cimport kick_particle as kick
from deposit cimport deposit_particle
from particle_boundary cimport periodic_x_cdef as periodic_x
from particle_boundary cimport calculate_ihole_cdef as calculate_ihole
from libc.math cimport fabs

def push_and_deposit(
         particle_t[:] particles, real3_t[:, :] E, real3_t[:, :] B,
         real_t qtmh, real_t dt, grid_t grid, int[:] ihole,
         real4_t[:,:] current, real_t S, const bint update):

    # Number of particles
    cdef int Np = particles.shape[0]

    # Electric and magnetic fields at particle location
    cdef real3_t e, b

    # A single particle struct
    cdef particle_t particle

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    # Variable needed by calculate_ihole
    cdef int ih = 0

    # Offset in interpolation for E and B-fields
    cdef real2_t offsetE, offsetB
    offsetB.x = grid.lbx
    offsetB.y = grid.lby - grid.noff
    offsetE.x = offsetB.x - 0.5
    offsetE.y = offsetB.y - 0.5

    # Because the particle position is stored in units of the grid spacing,
    # the drift velocity needs to be scaled by the grid spacing. This is
    # achieved easily by rescaling the time step.
    cdef real2_t dtds2
    dtds2.x = 0.5*dt/grid.dx
    dtds2.y = 0.5*dt/grid.dy

    # TODO: Define this in types.pxd
    cdef real_t Lx = <real_t> grid.nx

    for ip in range(Np):
        # Copy particle data to temporary struct
        particle = particles[ip]

        # Gather and electric & magnetic fields
        gather_cic(particles[ip], E, &e, offsetE)
        gather_cic(particles[ip], B, &b, offsetB)

        # Rescale values with qtmh = 0.5*dt*charge/mass
        rescale(&e, qtmh)
        rescale(&b, qtmh)

        # Kick the particle velocities
        kick(&particle, e, b)

        # First half of particle drift
        drift(&particle, dtds2)

        # Make sure that particle has not moved more than a half grid cell
        if (fabs(particle.x - particles[ip].x) > 0.5) or \
           (fabs(particle.y - particles[ip].y) > 0.5):
            ihole[0] = -1

        # Deposit the particle
        deposit_particle(particle, current, grid, S, offsetE)

        if update:
            # Second half of particle drift
            drift(&particle, dtds2)

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
