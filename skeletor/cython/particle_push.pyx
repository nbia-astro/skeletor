from types cimport real_t, real2_t, particle_t, grid_t
from cython.parallel import prange

def boris_push(particle_t[:] particles, real2_t[:, :] E,
                      real2_t[:, :] B, real_t qtmh, real_t dt, grid_t grid):

    cdef int Np = particles.shape[0]
    # Electric and magnetic fields at particle location
    cdef real2_t e, b

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    for ip in range(Np):
        # Gather and electric & magnetic fields with qtmh = 0.5*dt*charge/mass
        gather_cic(particles[ip], E, &e, grid, qtmh)
        gather_cic(particles[ip], B, &b, grid, qtmh)

        kick(&particles[ip], e, b)
        drift2(&particles[ip], dt, grid)

def modified_boris_push(particle_t[:] particles, real2_t[:, :] E,
                        real2_t[:, :] B, real_t qtmh, real_t dt, grid_t grid,
                        real_t Omega, real_t S):

    cdef int Np = particles.shape[0]
    # Electric and magnetic fields at particle location
    cdef real2_t e, b

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    for ip in range(Np):
        # Gather and electric & magnetic fields with qtmh = 0.5*dt*charge/mass
        gather_cic(particles[ip], E, &e, grid, qtmh)
        gather_cic(particles[ip], B, &b, grid, qtmh)

        # Modify fields due to rotation and shear
        b.z = b.z + Omega*dt
        e.y = e.y - S*particles[ip].y*b.z

        kick(&particles[ip], e, b)
        drift2(&particles[ip], dt, grid)


cdef inline void gather_cic(particle_t particle, real2_t[:,:] F, real2_t *f,
                        grid_t grid, real_t qtmh) nogil:

    cdef int ix, iy
    cdef real_t tx, ty, dx, dy
    cdef real_t x, y

    x = particle.x + grid.lbx - 0.5
    y = particle.y + grid.lby - 0.5 - grid.noff

    ix = <int> x
    iy = <int> y

    dx = x - <real_t> ix
    dy = y - <real_t> iy

    tx = 1.0 - dx
    ty = 1.0 - dy

    f.x  = dy*(dx*F[iy+1, ix+1].x + tx*F[iy+1, ix].x)  \
         + ty*(dx*F[iy  , ix+1].x + tx*F[iy  , ix].x)
    f.y  = dy*(dx*F[iy+1, ix+1].y + tx*F[iy+1, ix].y)  \
         + ty*(dx*F[iy  , ix+1].y + tx*F[iy  , ix].y)
    f.z  = dy*(dx*F[iy+1, ix+1].z + tx*F[iy+1, ix].z)  \
         + ty*(dx*F[iy  , ix+1].z + tx*F[iy  , ix].z)

    f.x = f.x*qtmh
    f.y = f.y*qtmh
    f.z = f.z*qtmh

cdef inline void kick(particle_t *particle, real2_t e, real2_t b) nogil:

    cdef real_t fac, vpx, vpy, vpz, vmx, vmy, vmz

    vmx = particle.vx + e.x
    vmy = particle.vy + e.y
    vmz = particle.vz + e.z

    vpx = vmx + (vmy*b.z - vmz*b.y)
    vpy = vmy + (vmz*b.x - vmx*b.z)
    vpz = vmz + (vmx*b.y - vmy*b.x)

    fac = 2./(1. + b.x*b.x + b.y*b.y + b.z*b.z)

    particle.vx = vmx + fac*(vpy*b.z - vpz*b.y) + e.x
    particle.vy = vmy + fac*(vpz*b.x - vpx*b.z) + e.y
    particle.vz = vmz + fac*(vpx*b.y - vpy*b.x) + e.z

cdef inline void drift2(particle_t *particle, real_t dt, grid_t grid) nogil:

    particle.x = particle.x + particle.vx*dt/grid.dx
    particle.y = particle.y + particle.vy*dt/grid.dy


def drift(particle_t[:] particles, real_t dt):

    cdef int Np = particles.shape[0]
    cdef int ip
    for ip in prange(Np, nogil=True, schedule='static'):

        particles[ip].x = particles[ip].x + particles[ip].vx*dt
        particles[ip].y = particles[ip].y + particles[ip].vy*dt
