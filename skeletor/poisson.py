class Poisson:

    """Solve Gauss' law ∇·E = ρ/ε0 via a discrete fourier transform."""

    def __init__(self, grid, ax, ay, np):

        from .cython.dtypes import Complex, Complex2, Int
        from .cython.ppic2_wrapper import cwpfft2rinit, cppois22
        from math import log2
        from numpy import zeros

        self.indx = int(log2(grid.nx))
        self.indy = int(log2(grid.ny))

        assert grid.nx == 2**self.indx, "'nx' needs to be a power of two"
        assert grid.ny == 2**self.indy, "'ny' needs to be a power of two"

        # Smoothed particle size in x- and y-direction
        self.ax = ax
        self.ay = ay

        # Normalization constant
        self.affp = grid.nx*grid.ny/np

        nxh = grid.nx//2
        nyh = (1 if 1 > grid.ny//2 else grid.ny//2)
        nxhy = (nxh if nxh > grid.ny else grid.ny)
        nxyh = (grid.nx if grid.nx > grid.ny else grid.ny)//2
        nye = grid.ny + 2
        kxp = (nxh - 1)//grid.nvp + 1
        kyp = (grid.ny - 1)//grid.nvp + 1

        self.qt = zeros((kxp, nye), Complex)
        self.fxyt = zeros((kxp, nye), Complex2)
        self.mixup = zeros(nxhy, Int)
        self.sct = zeros(nxyh, Complex)
        self.ffc = zeros((kxp, nyh), Complex)
        self.bs = zeros((kyp, kxp), Complex2)
        self.br = zeros((kyp, kxp), Complex2)

        # Prepare fft tables
        cwpfft2rinit(self.mixup, self.sct, self.indx, self.indy)

        # Calculate form factors
        isign = 0
        cppois22(
                self.qt, self.fxyt, isign, self.ffc,
                self.ax, self.ay, self.affp, grid)

    def __call__(self, qe, fxye, destroy_input=True):

        from .cython.ppic2_wrapper import cppois22, cwppfft2r, cwppfft2r2

        grid = qe.grid

        if destroy_input:
            qe_ = qe
        else:
            qe_ = qe.copy()

        # Transform charge to fourier space with standard procedure:
        # updates qt, modifies qe
        isign = -1
        ttp = cwppfft2r(
                qe_, self.qt, self.bs, self.br, isign, self.mixup, self.sct,
                self.indx, self.indy, grid)

        # Calculate force/charge in fourier space with standard procedure:
        # updates fxyt, we
        isign = -1
        we = cppois22(
                self.qt, self.fxyt, isign, self.ffc,
                self.ax, self.ay, self.affp, grid)

        # Transform force to real space with standard procedure:
        # updates fxye, modifies fxyt
        isign = 1
        cwppfft2r2(
                fxye, self.fxyt, self.bs, self.br, isign,
                self.mixup, self.sct, self.indx, self.indy, grid)

        return ttp, we