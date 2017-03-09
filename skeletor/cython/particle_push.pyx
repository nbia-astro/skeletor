from types cimport real_t, real3_t, particle_t, grid_t
from cython.parallel import prange

def boris_push(particle_t[:] particles, real3_t[:, :] E,
                      real3_t[:, :] B, real_t qtmh, real_t dt, grid_t grid):

    cdef int Np = particles.shape[0]
    # Electric and magnetic fields at particle location
    cdef real3_t e, b

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    # Offset in interpolation for E and B-fields
    cdef real3_t offsetE, offsetB
    offsetB.x = grid.lbx
    offsetB.y = grid.lby - grid.noff
    offsetE.x = offsetB.x - 0.5
    offsetE.y = offsetB.y - 0.5

    for ip in range(Np):
        # Gather and electric & magnetic fields
        gather_cic(particles[ip], E, &e, offsetE)
        gather_cic(particles[ip], B, &b, offsetB)

        # Rescale values with qtmh = 0.5*dt*charge/mass
        rescale(&e, qtmh)
        rescale(&b, qtmh)

        kick(&particles[ip], e, b)
        drift2(&particles[ip], dt, grid)

def modified_boris_push(particle_t[:] particles, real3_t[:, :] E,
                        real3_t[:, :] B, real_t qtmh, real_t dt, grid_t grid,
                        real_t Omega, real_t S):

    cdef int Np = particles.shape[0]
    # Electric and magnetic fields at particle location
    cdef real3_t e, b

    # It might be better to use `Py_ssize_t` instead of `int`
    cdef int ip

    # Offset in interpolation for E and B-fields
    cdef real3_t offsetE, offsetB
    offsetB.x = grid.lbx
    offsetB.y = grid.lby - grid.noff
    offsetE.x = offsetB.x - 0.5
    offsetE.y = offsetB.y - 0.5

    for ip in range(Np):
        # Gather and electric & magnetic fields
        gather_cic(particles[ip], E, &e, offsetE)
        gather_cic(particles[ip], B, &b, offsetB)

        # Rescale values with qtmh = 0.5*dt*charge/mass
        rescale(&e, qtmh)
        rescale(&b, qtmh)

        # Modify fields due to rotation and shear
        b.z = b.z + Omega*dt
        # TODO: We need to get this working/Make this more general
        e.y = e.y - S*particles[ip].y*grid.dy*b.z

        kick(&particles[ip], e, b)
        drift2(&particles[ip], dt, grid)

cdef inline void gather_cic(particle_t particle, real3_t[:,:] F, real3_t *f,
                        real3_t offset) nogil:

    cdef int ix, iy
    cdef real_t tx, ty, dx, dy
    cdef real_t x, y

    x = particle.x + offset.x
    y = particle.y + offset.y

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

cdef inline void rescale(real3_t *f, real_t qtmh) nogil:

    f.x = f.x*qtmh
    f.y = f.y*qtmh
    f.z = f.z*qtmh

cdef inline void kick(particle_t *particle, real3_t e, real3_t b) nogil:

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
