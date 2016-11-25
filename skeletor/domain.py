class Domain:
    """
    This class defines the global computational domain. It does not know about
    differential operators, interpolation, or parallelization. Hence there is
    no need to define guard layers.
    """

    def __init__(self, nx, ny):

        # Number of grid points in x- and y-direction
        self.nx = nx
        self.ny = ny

    @property
    def Lx(self):
        "Domain size in x"
        # This is defined as a property because right now,
        # Lx must be equal to nx.
        return self.nx

    @property
    def Ly(self):
        "Domain size in y"
        # This is defined as a property because right now,
        # Ly must be equal to ny.
        return self.ny

    @property
    def dx(self):
        "Grid spacing in x"
        return self.Lx/self.nx

    @property
    def dy(self):
        "Grid spacing in y"
        return self.Ly/self.ny

    @property
    def x(self):
        "One-dimensional x-coordinate array"
        from numpy import arange
        return arange(self.nx)

    @property
    def y(self):
        "One-dimensional y-coordinate array"
        from numpy import arange
        return arange(self.ny)


class SubDomain(Domain):

    def __init__(self, nx, ny, comm):

        from .cython.ppic2_wrapper import cppinit
        from numpy import array

        # Initialize Domain class
        super().__init__(nx, ny)

        # MPI communicator
        self.comm = comm

        # Start parallel processing. Calling this function necessary even
        # though `MPI.Init()` has already been called by importing `MPI` from
        # `mpi4py`. The reason is that `cppinit()` sets a number of global
        # variables in the C library source file (`ppic2/pplib2.c`). The
        # returned variables `idproc` and `nvp` are simply the MPI rank (i.e.
        # processor id) and size (i.e. total number of processes),
        # respectively.
        idproc, nvp = cppinit(comm)

        # nyp = number of primary (complete) gridpoints in particle partition
        self.nyp = ny//comm.size

        # noff = lowermost global gridpoint in particle partition
        self.noff = self.nyp*comm.rank

        # edges[0:1] = lower:upper boundary of particle partition
        self.edges = array([self.noff, self.noff + self.nyp])

        # Parameters needed by PPIC2
        self.kstrt = comm.rank + 1
        self.nvp = comm.size

        if comm.size > self.ny:
            msg = "Too many processors requested: ny={}, comm.size={}"
            raise RuntimeError(msg.format(self.ny, comm.size))

    @property
    def y(self):
        "One-dimensional y-coordinate array"
        from numpy import arange
        return arange(self.noff, self.noff + self.nyp)

    @property
    def yg(self):
        "One-dimensional y-coordinate array including ghost"
        from numpy import arange
        return arange(self.noff, self.noff + self.nyp + 1)
