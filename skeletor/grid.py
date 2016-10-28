from .cython.ppic2_wrapper import grid_t


class Grid(grid_t):

    def __init__(self, nx, ny, comm, nlbx=0, nubx=2, nlby=0, nuby=1):

        """"""
        from numpy import array

        # Number of grid points in x- and y-direction
        self.nx = nx
        self.ny = ny

        # nyp = number of primary (complete) gridpoints in particle partition
        self.nyp = ny//comm.size

        # noff = lowermost global gridpoint in particle partition
        self.noff = self.nyp*comm.rank

        # edges[0:1] = lower:upper boundary of particle partition
        self.edges = array([self.noff, self.noff + self.nyp])

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

        # nypmx = size of particle partition, including guard cells
        # nypmn = value of nyp
        self.nypmx = self.nyp + self.nlby + self.nuby
        self.nypmn = self.nyp

        if comm.size > self.ny:
            msg = "Too many processors requested: ny={}, comm.size={}"
            raise RuntimeError(msg.format(self.ny, comm.size))

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
    def x(self):
        "One-dimensional x-coordinate array"
        from numpy import arange
        return arange(self.nx)

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
