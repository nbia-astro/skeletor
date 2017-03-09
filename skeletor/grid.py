from .cython.types import grid_t


class Grid(grid_t):

    def __init__(self, nx, ny, comm, lbx=1, lby=1, Lx=1.0, Ly=1.0):

        from .cython.ppic2_wrapper import cppinit

        # Number of grid points in x- and y-direction
        self.nx = nx
        self.ny = ny

        # Grid size in x- and y-direction
        self.Lx = Lx
        self.Ly = Ly

        # Grid cell size
        self.dx = self.Lx/self.nx
        self.dy = self.Ly/self.ny

        # MPI communicator
        self.comm = comm

        # Start parallel processing. Calling this function necessary even
        # though `MPI.Init()` has already been called by importing `MPI` from
        # `mpi4py`. The reason is that `cppinit()` sets a number of global
        # variables in the C library source file (`ppic2/pplib2.c`). The
        # returned variables `idproc` and `nvp` are simply the MPI rank (i.e.
        # processor id) and size (i.e. total number of processes),
        # respectively.
        # TODO: Move this somewhere else. The constructor of the grid class is
        # not exactly the obvious place for initializing MPI at C-level.
        idproc, nvp = cppinit(comm)

        # nyp = number of primary (complete) gridpoints in particle partition
        self.nyp = ny//comm.size

        # noff = lowermost global gridpoint in particle partition
        self.noff = self.nyp*comm.rank

        # edges[0:1] = lower:upper boundary of particle partition
        # Note that self.edges is always a floating point list as defined in
        # the Cython extension type "grid_t" that this class inherits from.
        self.edges = [self.noff, self.noff + self.nyp]

        # lbx, lby = first active index in x, y
        # ubx, uby = index of first ghost upper zone in x, y
        self.lbx = lbx
        self.ubx = lbx + self.nx
        self.lby = lby
        self.uby = lby + self.nyp

        # mx and myp are the total (active plus guard) number of grid points in
        # x and y in each subdomain
        self.mx = self.nx + 2*self.lbx
        self.myp = self.nyp + 2*self.lby

        if comm.size > self.ny:
            msg = "Too many processors requested: ny={}, comm.size={}"
            raise RuntimeError(msg.format(self.ny, comm.size))

    @property
    def x(self):
        "One-dimensional x-coordinate array"
        from numpy import arange
        return (arange(self.nx) + 0.5)*self.dx

    @property
    def y(self):
        "One-dimensional y-coordinate array"
        from numpy import arange
        return (arange(self.noff, self.noff + self.nyp) + 0.5)*self.dy

    @property
    def yg(self):
        "One-dimensional y-coordinate array including ghost"
        from numpy import arange
        return (arange(self.noff - self.lby, self.noff + self.nyp + self.lby)
                + 0.5)*self.dy
