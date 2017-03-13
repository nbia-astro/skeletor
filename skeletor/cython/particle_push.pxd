from types cimport real_t, real2_t, real3_t, particle_t, grid_t

cdef inline void gather_cic(particle_t particle, real3_t[:,:] F, real3_t *f,
                        real2_t offset) nogil:

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

cdef inline void kick_particle(particle_t *particle,
                               real3_t e, real3_t b) nogil:

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

cdef inline void drift_particle(particle_t *particle, real2_t dtds) nogil:

    particle.x = particle.x + particle.vx*dtds.x
    particle.y = particle.y + particle.vy*dtds.y

cdef inline void rescale(real3_t *f, real_t qtmh) nogil:

    f.x = f.x*qtmh
    f.y = f.y*qtmh
    f.z = f.z*qtmh
