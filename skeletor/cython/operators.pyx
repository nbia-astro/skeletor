from ctypes cimport complex_t, complex2_t, float_t, float2_t
from libc.math cimport M_PI
cimport cython

cdef extern from "complex.h":
    float_t crealf(complex_t) nogil
    float_t cimagf(complex_t) nogil
    complex_t conjf(complex_t) nogil
    complex_t _Complex_I

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef float_t grad_inv_del(
        complex_t[:,:] qt, complex2_t[:,:] fxyt, complex_t[:,:] ffc,
        int nx, int ny, int kstrt) nogil:
    """
    This function computes E=∇(∇⁻²ρ), where ∇⁻² is the inverse Laplacian.
    E thus satisfies Gauss' law ∇·E=ρ.

    The calculation is done in Fourier space.
    
    Note: Right now this is just a reimplementation PPIC2's cppois22 (see
    ppic2_wrapper.pyx). That's why the "energy" is also computed and returned.
    """

    cdef int kxp = qt.shape[0]
    cdef double wp = 0.0

    cdef int nxh, nyh, ks, joff, kxps, j, k, k1
    cdef float_t dnx, dny, dkx, dky, at1, at2, at3
    cdef complex_t zero, zt1, zt2

    nxh = nx/2
    nyh = 1 if 1 > ny/2 else ny/2
    ks = kstrt - 1
    joff = kxp*ks
    kxps = nxh - joff
    kxps = 0 if 0 > kxps else kxps
    kxps = kxp if kxp < kxps else kxps
    dnx = 2.0*M_PI/<float_t> nx
    dny = 2.0*M_PI/<float_t> ny
    zero = 0.0 + 0.0*_Complex_I

    if kstrt > nxh:
        return 0.0

    # mode numbers 0 < kx < nx/2 and 0 < ky < ny/2
    for j in range(kxps):
        dkx = dnx*<float_t> (j + joff)
        if j + joff > 0:
            for k in range(1, nyh):
                k1 = ny - k
                at1 = crealf(ffc[j, k])*cimagf(ffc[j, k])
                at2 = dkx*at1
                at3 = dny*at1*<float_t> k
                zt1 = cimagf(qt[j, k]) - crealf(qt[j, k])*_Complex_I
                zt2 = cimagf(qt[j, k1]) - crealf(qt[j, k1])*_Complex_I
                fxyt[j, k].x = at2*zt1
                fxyt[j, k].y = at3*zt1
                fxyt[j, k1].x = at2*zt2
                fxyt[j, k1].y = -at3*zt2
                wp += at1*crealf(qt[j, k]*conjf(qt[j, k]) \
                               + qt[j, k1]*conjf(qt[j, k1]))
            # mode numbers ky = 0, ny/2
            at1 = crealf(ffc[j, 0])*cimagf(ffc[j, 0])
            at3 = dkx*at1
            zt1 = cimagf(qt[j, 0]) - crealf(qt[j, 0])*_Complex_I
            fxyt[j, 0].x = at3*zt1
            fxyt[j, 0].y = zero
            fxyt[j, nyh].x = zero
            fxyt[j, nyh].y = zero
            wp += at1*crealf(qt[j, 0]*conjf(qt[j, 0]))
    # mode numbers kx = 0, nx/2
    if ks == 0:
        for k in range(1, nyh):
            k1 = ny - k
            at1 = crealf(ffc[0, k])*cimagf(ffc[0, k])
            at2 = dny*at1*<float_t> k
            zt1 = cimagf(qt[0, k]) - crealf(qt[0, k])*_Complex_I
            fxyt[0, k].x = zero
            fxyt[0, k].y = at2*zt1
            fxyt[0, k1].x = zero
            fxyt[0, k1].y = zero
            wp += at1*crealf(qt[0, k]*conjf(qt[0, k]))
        fxyt[0, 0].x = zero;
        fxyt[0, 0].y = zero;
        fxyt[0, nyh].x = zero;
        fxyt[0, nyh].y = zero;

    return wp*(<float_t> nx)*(<float_t> ny)
