class Poisson:

    """Solve Gauss' law ∇·E = ρ/ε0 via a discrete fourier transform."""

    def __init__(self, grid, comm, ax, ay, np):

        from dtypes import Complex, Complex2, Int
        from ppic2_wrapper import cwpfft2rinit, cppois22
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

        # kstrt = comm.rank + 1
        nvp = comm.size

        nxh = grid.nx//2
        nyh = (1 if 1 > grid.ny//2 else grid.ny//2)
        nxhy = (nxh if nxh > grid.ny else grid.ny)
        nxyh = (grid.nx if grid.nx > grid.ny else grid.ny)//2
        nye = grid.ny + 2
        kxp = (nxh - 1)//nvp + 1
        kyp = (grid.ny - 1)//nvp + 1

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
                self.ax, self.ay, self.affp, grid, comm)

    def __call__(self, qe, fxye):

        from ppic2_wrapper import cppois22, cwppfft2r, cwppfft2r2

        grid = qe.grid
        comm = qe.comm

        # Transform charge to fourier space with standard procedure:
        # updates qt, modifies qe
        isign = -1
        ttp = cwppfft2r(
                qe.copy(), self.qt, self.bs, self.br, isign,
                self.mixup, self.sct, self.indx, self.indy, comm)

        # Calculate force/charge in fourier space with standard procedure:
        # updates fxyt, we
        isign = -1
        we = cppois22(
                self.qt, self.fxyt, isign, self.ffc,
                self.ax, self.ay, self.affp, grid, comm)

        # Transform force to real space with standard procedure:
        # updates fxye, modifies fxyt
        isign = 1
        cwppfft2r2(
                fxye, self.fxyt, self.bs, self.br, isign,
                self.mixup, self.sct, self.indx, self.indy, comm)

        return ttp, we


if __name__ == "__main__":

    from ppic2_wrapper import cppinit
    from dtypes import Float, Float2
    from grid import Grid
    from fields import Field
    from mpi4py.MPI import COMM_WORLD as comm

    import numpy
    import matplotlib.pyplot as plt

    # Spatial resolution
    indx, indy = 5, 5
    nx = 1 << indx
    ny = 1 << indy

    # Average number of particles per cell
    npc = 256

    # Smoothed particle size in x/y direction
    ax = 0.912871
    ay = 0.912871

    # Total number of particles
    np = nx*ny*npc

    #############################################
    # Solve Gauss' law with PPIC's parallel FFT #
    #############################################

    # Start parallel processing.
    idproc, nvp = cppinit(comm)

    # Create numerical grid
    grid = Grid(nx, ny, comm)

    # Initialize Poisson solver
    poisson = Poisson(grid, comm, ax, ay, np)

    # Coordinate arrays
    x = numpy.arange(grid.nx, dtype=Float)
    y = numpy.arange(grid.ny, dtype=Float)
    xx, yy = numpy.meshgrid(x, y)

    # Initialize density field
    qe = Field(grid, comm, dtype=Float)
    qe.fill(0.0)
    ikx, iky = 1, 2
    qe[:ny, :nx] = numpy.sin(2*numpy.pi*(ikx*xx/nx + iky*yy/ny))

    # Initialize force field
    fxye = Field(grid, comm, dtype=Float2)
    fxye.fill((0.0, 0.0))

    # Solve Gauss' law
    poisson(qe, fxye)

    ##############################################
    # Solve Gauss' law with Numpy's built-in FFT #
    ##############################################

    # Wave number arrays
    kx = 2*numpy.pi*numpy.fft.rfftfreq(grid.nx)
    ky = 2*numpy.pi*numpy.fft.fftfreq(grid.ny)
    kx, ky = numpy.meshgrid(kx, ky)

    # Normalization constant
    affp = grid.nx*grid.ny/np

    # Compute inverse wave number squared
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    k21 = 1.0/k2
    k21[0, 0] = 0.0
    k2[0, 0] = 0.0

    # Effective inverse wave number for finite size particles
    # TODO: Figure out how the exponential factor is actually derived
    k21_eff = k21*numpy.exp(-((kx*ax)**2 + (ky*ay)**2))

    # Transform charge density to Fourier space
    qt = numpy.fft.rfft2(qe[:ny, :nx])

    # Solve Gauss' law in Fourier space and transform back to real space
    fx = affp*numpy.fft.irfft2(-1j*kx*k21_eff*qt)
    fy = affp*numpy.fft.irfft2(-1j*ky*k21_eff*qt)

    # Make sure the two solutions are close to each other
    assert numpy.allclose(fx, fxye[:ny, :nx]["x"])
    assert numpy.allclose(fy, fxye[:ny, :nx]["y"])

    #############
    # Visualize #
    #############

    plt.rc('image', origin='lower', interpolation='nearest')
    plt.figure(1)
    plt.clf()
    ax1 = plt.subplot2grid((2, 4), (0, 1), colspan=2)
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    ax1.imshow(qe)
    ax2.imshow(fxye["x"])
    ax3.imshow(fxye["y"])
    ax1.set_title(r'$\rho$')
    ax2.set_title(r'$f_x$')
    ax3.set_title(r'$f_y$')
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
    plt.draw()
    plt.show()
