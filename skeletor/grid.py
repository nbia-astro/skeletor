from .cython.types import grid_t


class Grid(grid_t):

    def __init__(self, nx, ny, comm, nlbx=0, nubx=2, nlby=0, nuby=1,
                 Lx=None, Ly=None):

        from .cython.ppic2_wrapper import cppinit

        # Number of grid points in x- and y-direction
        self.nx = nx
        self.ny = ny

        # Grid size in x- and y-direction
        self.Lx = Lx if Lx is not None else nx
        self.Ly = Ly if Ly is not None else ny

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
        idproc, nvp = cppinit(comm)

        # nyp = number of primary (complete) gridpoints in particle partition
        self.nyp = ny//comm.size

        # noff = lowermost global gridpoint in particle partition
        self.noff = self.nyp*comm.rank

        # edges[0:1] = lower:upper boundary of particle partition
        # Note that self.edges is always a floating point list as defined in
        # the Cython extension type "grid_t" that this class inherits from.
        self.edges = [self.noff, self.noff + self.nyp]

        # Ghost zone setup
        # nlbx, nlby = number of ghost zones at lower boundary in x, y
        # nubx, nuby = number of ghost zones at upper boundary in x, y
        self.nlbx = nlbx
        self.nlby = nlby
        self.nubx = nubx
        self.nuby = nuby

        # lbx, lby = first active index in x, y
        # ubx, uby = index of first ghost upper zone in x, y
        self.lbx = nlbx
        self.ubx = self.nx + self.nlbx
        self.lby = nlby
        self.uby = self.nyp + self.nlby

        # Parameters needed by PPIC2
        self.kstrt = comm.rank + 1
        self.nvp = comm.size

        # nypmx = size of particle partition, including guard cells, in y
        # nxpmx = size of particle partition, including guard cells, in x
        # nypmn = value of nyp
        self.nypmx = self.nyp + self.nlby + self.nuby
        self.nypmn = self.nyp
        self.nxpmx = self.nx + self.nlbx + self.nubx

        if comm.size > self.ny:
            msg = "Too many processors requested: ny={}, comm.size={}"
            raise RuntimeError(msg.format(self.ny, comm.size))

    @property
    def x(self):
        "One-dimensional x-coordinate array"
        from numpy import arange
        return arange(self.nx) + 0.5

    @property
    def y(self):
        "One-dimensional y-coordinate array"
        from numpy import arange
        return arange(self.noff, self.noff + self.nyp) + 0.5

    @property
    def yg(self):
        "One-dimensional y-coordinate array including ghost"
        from numpy import arange
        return arange(self.noff - self.lby,
                      self.noff + self.nyp + self.nuby) + 0.5
