from ..grid import Grid


class Manifold(Grid):

    def __init__(
            self, nx, ny, comm,
            ax=0.0, ay=0.0, nlbx=0, nubx=2, nlby=0, nuby=1):

        from ..cython.dtypes import Complex, Complex2, Float, Float2, Int
        from ..cython.ppic2_wrapper import cwpfft2rinit, cppois22
        from math import log2
        from numpy import zeros

        super().__init__(
                nx, ny, comm, nlbx=nlbx, nubx=nubx, nlby=nlby, nuby=nuby)

        self.indx = int(log2(nx))
        self.indy = int(log2(ny))

        assert nx == 2**self.indx, "'nx' needs to be a power of two"
        assert ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Normalization constant
        self.affp = 1.0

        nxh = nx//2
        nyh = (1 if 1 > ny//2 else ny//2)
        nxhy = (nxh if nxh > ny else ny)
        nxyh = (nx if nx > ny else ny)//2
        nye = ny + 2
        kxp = (nxh - 1)//self.comm.size + 1
        kyp = (ny - 1)//self.comm.size + 1

        self.qt = zeros((kxp, nye), Complex)
        self.fxyt = zeros((kxp, nye), Complex2)
        self.mixup = zeros(nxhy, Int)
        self.sct = zeros(nxyh, Complex)
        self.ffc = zeros((kxp, nyh), Complex)
        self.bs = zeros((kyp, kxp), Complex2)
        self.br = zeros((kyp, kxp), Complex2)

        # Declare charge and electric fields with fixed number of guard layers.
        # This is necessary for interfacing with PPIC2's C-routines.
        self.qe = zeros((self.nyp+1, nx+2), dtype=Float)
        self.fxye = zeros((self.nyp+1, nx+2), dtype=Float2)

        # Prepare fft tables
        cwpfft2rinit(self.mixup, self.sct, self.indx, self.indy)

        # Calculate form factors
        isign = 0
        cppois22(
                self.qt, self.fxyt, isign, self.ffc,
                self.ax, self.ay, self.affp, self)

    def gradient(self, qe, fxye):

        from ..cython.ppic2_wrapper import cwppfft2r, cwppfft2r2
        from ..cython.operators import grad

        # Copy charge into pre-allocated buffer that has the right number of
        # guard layers expected by PPIC2
        self.qe[:-1, :-2] = qe.active

        # Transform charge to fourier space with standard procedure:
        # updates qt, modifies qe
        isign = -1
        ttp = cwppfft2r(self.qe, self.qt, self.bs, self.br, isign,
                        self.mixup, self.sct, self.indx, self.indy, self)

        # Calculate gradient in fourier space
        # updates fxyt
        kstrt = self.comm.rank + 1
        grad(self.qt, self.fxyt, self.ffc, self.affp, self.nx, self.ny, kstrt)

        # Transform force to real space with standard procedure:
        # updates fxye, modifies fxyt
        isign = 1
        cwppfft2r2(self.fxye, self.fxyt, self.bs, self.br, isign,
                   self.mixup, self.sct, self.indx, self.indy, self)

        # Copy electric field into an array with arbitrary number
        # of guard layers
        fxye.active = self.fxye[:-1, :-2]

        return ttp

    def log(self, f):
        """Custom log function that works on the
            active cells of skeletor fields"""
        from numpy import log as numpy_log
        g = f.copy()
        g[self.lby:self.uby, self.lbx:self.ubx] = numpy_log(f.active)
        return g

    def grad_inv_del(self, qe, fxye, custom_cppois22=False):

        from ..cython.ppic2_wrapper import cppois22, cwppfft2r, cwppfft2r2
        from ..cython.operators import grad_inv_del

        # Copy charge into pre-allocated buffer that has the right number of
        # guard layers expected by PPIC2
        self.qe[:-1, :-2] = qe.active

        # Transform charge to fourier space with standard procedure:
        # updates qt, modifies qe
        isign = -1
        ttp = cwppfft2r(self.qe, self.qt, self.bs, self.br, isign,
                        self.mixup, self.sct, self.indx, self.indy, self)

        # Calculate force/charge in fourier space with standard procedure:
        # updates fxyt, we
        if custom_cppois22:
            kstrt = self.comm.rank + 1
            we = grad_inv_del(
                    self.qt, self.fxyt, self.ffc, self.nx, self.ny, kstrt)
        else:
            isign = -1
            we = cppois22(
                    self.qt, self.fxyt, isign, self.ffc,
                    self.ax, self.ay, self.affp, self)

        # Transform force to real space with standard procedure:
        # updates fxye, modifies fxyt
        isign = 1
        cwppfft2r2(self.fxye, self.fxyt, self.bs, self.br, isign,
                   self.mixup, self.sct, self.indx, self.indy, self)

        # Copy electric field into an array with arbitrary number
        # of guard layers
        fxye.active = self.fxye[:-1, :-2]

        return ttp, we
