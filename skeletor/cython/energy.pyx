from types cimport real_t, particle_t


def kinetic_energy(particle_t[:] particles):
    cdef int Np = particles.shape[0]
    cdef int ip

    cdef real_t ekin = 0.0

    for ip in range(Np):
        ekin = ekin + particles[ip].vx*particles[ip].vx
        ekin = ekin + particles[ip].vy*particles[ip].vy
        ekin = ekin + particles[ip].vz*particles[ip].vz

    ekin = 0.5*ekin

    return ekin
