from ctypes cimport real_t, particle_t

def push_epicycle(particle_t[:] particles, real_t dt):

    cdef real_t ey, bz

    cdef real_t vmx, vmy
    cdef real_t vpx, vpy
    cdef real_t fac

    cdef real_t Omega = 1.0
    cdef real_t S = -1.5

    for ip in range(particles.shape[0]):

        bz = Omega*dt
        ey = -S*particles[ip].y*bz

        vmx = particles[ip].vx
        vmy = particles[ip].vy + ey

        vpx = vmx + vmy*bz
        vpy = vmy - vmx*bz

        fac = 2.0/(1.0 + bz*bz)

        particles[ip].vx = vmx + fac*vpy*bz
        particles[ip].vy = vmy - fac*vpx*bz + ey

        particles[ip].x += particles[ip].vx*dt
        particles[ip].y += particles[ip].vy*dt
