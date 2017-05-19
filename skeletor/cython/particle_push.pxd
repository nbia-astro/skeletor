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

cdef inline void gather_tsc(particle_t particle, real3_t[:,:] F, real3_t *f,
                        real2_t offset) nogil:

    cdef int ix, iy
    cdef real_t dx, dy
    cdef real_t x, y
    cdef real_t wmx, w0x, wpx, wmy, w0y, wpy

    x = particle.x + offset.x
    y = particle.y + offset.y

    x = x + 0.5
    y = y + 0.5

    ix = <int> x
    iy = <int> y

    dx = x - <real_t> ix - 0.5
    dy = y - <real_t> iy - 0.5

    w0x = 0.75 - dx*dx
    wpx = 0.5*(0.5 + dx)**2
    wmx = 1.0 - (w0x + wpx)

    w0y = 0.75 - dy*dy
    wpy = 0.5*(0.5 + dy)**2
    wmy = 1.0 - (w0y + wpy)

    f.x = wmy*(wmx*F[iy-1, ix-1].x+w0x*F[iy-1, ix  ].x+wpx*F[iy-1, ix+1].x)\
        + w0y*(wmx*F[iy  , ix-1].x+w0x*F[iy  , ix  ].x+wpx*F[iy  , ix+1].x)\
        + wpy*(wmx*F[iy+1, ix-1].x+w0x*F[iy+1, ix  ].x+wpx*F[iy+1, ix+1].x)

    f.y = wmy*(wmx*F[iy-1, ix-1].y+w0x*F[iy-1, ix  ].y+wpx*F[iy-1, ix+1].y)\
        + w0y*(wmx*F[iy  , ix-1].y+w0x*F[iy  , ix  ].y+wpx*F[iy  , ix+1].y)\
        + wpy*(wmx*F[iy+1, ix-1].y+w0x*F[iy+1, ix  ].y+wpx*F[iy+1, ix+1].y)

    f.z = wmy*(wmx*F[iy-1, ix-1].z+w0x*F[iy-1, ix  ].z+wpx*F[iy-1, ix+1].z)\
        + w0y*(wmx*F[iy  , ix-1].z+w0x*F[iy  , ix  ].z+wpx*F[iy  , ix+1].z)\
        + wpy*(wmx*F[iy+1, ix-1].z+w0x*F[iy+1, ix  ].z+wpx*F[iy+1, ix+1].z)

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
