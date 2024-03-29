from types cimport complex_t, complex2_t, real_t, real3_t, grid_t
from libc.math cimport M_PI, exp, pow
cimport cython

cdef extern from "complex.h":
    float crealf(float complex) nogil
    float cimagf(float complex) nogil
    float complex conjf(float complex) nogil
    float complex _Complex_I

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef void calc_form_factors(
        complex_t[:,:] qt, complex_t[:,:] ffc,
        real_t ax, real_t ay, real_t affp,
        grid_t grid, int kstrt) nogil:

    cdef int nx = grid.nx
    cdef int ny = grid.ny
    cdef int kxp = qt.shape[0]

    cdef int nxh, nyh, ks, joff, kxps, j, k
    cdef real_t dnx, dny, dkx, dky, at1, at2, at3, at4

    nxh = nx/2
    nyh = 1 if 1 > ny/2 else ny/2
    ks = kstrt - 1
    joff = kxp*ks
    kxps = nxh - joff
    kxps = 0 if 0 > kxps else kxps
    kxps = kxp if kxp < kxps else kxps
    dnx = 2.0*M_PI/grid.Lx
    dny = 2.0*M_PI/grid.Ly

    if kstrt > nxh:
        return

    # Prepare form factor array
    for j in range(kxps):
        dkx = dnx*<real_t> (j + joff)
        at1 = dkx*dkx;
        at2 = pow((dkx*ax),2)
        for k in range(nyh):
            dky = dny*<real_t> k
            at3 = dky*dky + at1;
            at4 = exp(-.5*(pow((dky*ay),2) + at2))
            if at3 == 0.0:
                ffc[j, k] = affp + 1.0*_Complex_I;
            else:
                ffc[j, k] = (affp*at4/at3) + at4*_Complex_I;

@cython.cdivision(True)
@cython.boundscheck(False)
cpdef real_t grad_inv_del(
        complex_t[:,:] qt, complex2_t[:,:] fxyt, complex_t[:,:] ffc,
        grid_t grid, int kstrt) nogil:
    """
    This function computes E=∇(∇⁻²ρ), where ∇⁻² is the inverse Laplacian.
    E thus satisfies Gauss' law ∇·E=ρ.

    The calculation is done in Fourier space.
    
    Note: Right now this is just a reimplementation PPIC2's cppois22 (see
    ppic2_wrapper.pyx). That's why the "energy" is also computed and returned.
    """

    cdef int nx = grid.nx
    cdef int ny = grid.ny
    cdef int kxp = qt.shape[0]
    cdef double wp = 0.0

    cdef int nxh, nyh, ks, joff, kxps, j, k, k1
    cdef real_t dnx, dny, dkx, dky, at1, at2, at3
    cdef complex_t zero, zt1, zt2

    nxh = nx/2
    nyh = 1 if 1 > ny/2 else ny/2
    ks = kstrt - 1
    joff = kxp*ks
    kxps = nxh - joff
    kxps = 0 if 0 > kxps else kxps
    kxps = kxp if kxp < kxps else kxps
    dnx = 2.0*M_PI/grid.Lx
    dny = 2.0*M_PI/grid.Ly
    zero = 0.0

    if kstrt > nxh:
        return 0.0

    # mode numbers 0 < kx < nx/2 and 0 < ky < ny/2
    for j in range(kxps):
        dkx = dnx*<real_t> (j + joff)
        if j + joff > 0:
            for k in range(1, nyh):
                dky = dny*<real_t> k
                k1 = ny - k
                at1 = crealf(ffc[j, k])*cimagf(ffc[j, k])
                at2 = dkx*at1
                at3 = dky*at1
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
            at2 = dkx*at1
            zt1 = cimagf(qt[j, 0]) - crealf(qt[j, 0])*_Complex_I
            fxyt[j, 0].x = at2*zt1
            fxyt[j, 0].y = zero
            fxyt[j, nyh].x = zero
            fxyt[j, nyh].y = zero
            wp += at1*crealf(qt[j, 0]*conjf(qt[j, 0]))
    # mode numbers kx = 0, nx/2
    if ks == 0:
        for k in range(1, nyh):
            dky = dny*<real_t> k
            k1 = ny - k
            at1 = crealf(ffc[0, k])*cimagf(ffc[0, k])
            at3 = dky*at1
            zt1 = cimagf(qt[0, k]) - crealf(qt[0, k])*_Complex_I
            fxyt[0, k].x = zero
            fxyt[0, k].y = at3*zt1
            fxyt[0, k1].x = zero
            fxyt[0, k1].y = zero
            wp += at1*crealf(qt[0, k]*conjf(qt[0, k]))
        fxyt[0, 0].x = zero
        fxyt[0, 0].y = zero
        fxyt[0, nyh].x = zero
        fxyt[0, nyh].y = zero

    return wp*(<real_t> nx)*(<real_t> ny)


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef void grad(
        complex_t[:,:] qt, complex2_t[:,:] fxyt, complex_t[:,:] ffc,
        real_t affp, int nx, int ny, int kstrt) nogil:
    """
    This function computes the gradient in Fourier space.
    """

    cdef int kxp = qt.shape[0]

    cdef int nxh, nyh, ks, joff, kxps, j, k, k1
    cdef real_t dnx, dny, dkx, dky, at1, at2, at3
    cdef complex_t zero, zt1, zt2

    nxh = nx/2
    nyh = 1 if 1 > ny/2 else ny/2
    ks = kstrt - 1
    joff = kxp*ks
    kxps = nxh - joff
    kxps = 0 if 0 > kxps else kxps
    kxps = kxp if kxp < kxps else kxps
    dnx = 2.0*M_PI/<real_t> nx
    dny = 2.0*M_PI/<real_t> ny
    zero = 0.0

    if kstrt > nxh:
        return

    # mode numbers 0 < kx < nx/2 and 0 < ky < ny/2
    for j in range(kxps):
        dkx = dnx*<real_t> (j + joff)
        if j + joff > 0:
            for k in range(1, nyh):
                dky = dny*<real_t> k
                k1 = ny - k
                at1 = affp*cimagf(ffc[j, k])
                at2 = dkx*at1
                at3 = dky*at1
                zt1 = -cimagf(qt[j, k]) + crealf(qt[j, k])*_Complex_I
                zt2 = -cimagf(qt[j, k1]) + crealf(qt[j, k1])*_Complex_I
                fxyt[j, k].x = at2*zt1
                fxyt[j, k].y = at3*zt1
                fxyt[j, k1].x = at2*zt2
                fxyt[j, k1].y = -at3*zt2
            # mode numbers ky = 0, ny/2
            at1 = affp*cimagf(ffc[j, 0])
            at2 = dkx*at1
            zt1 = -cimagf(qt[j, 0]) + crealf(qt[j, 0])*_Complex_I
            fxyt[j, 0].x = at2*zt1
            fxyt[j, 0].y = zero
            fxyt[j, nyh].x = zero
            fxyt[j, nyh].y = zero
    # mode numbers kx = 0, nx/2
    if ks == 0:
        for k in range(1, nyh):
            dky = dny*<real_t> k
            k1 = ny - k
            at1 = affp*cimagf(ffc[0, k])
            at3 = dky*at1
            zt1 = -cimagf(qt[0, k]) + crealf(qt[0, k])*_Complex_I
            fxyt[0, k].x = zero
            fxyt[0, k].y = at3*zt1
            fxyt[0, k1].x = zero
            fxyt[0, k1].y = zero
        fxyt[0, 0].x = zero
        fxyt[0, 0].y = zero
        fxyt[0, nyh].x = zero
        fxyt[0, nyh].y = zero
