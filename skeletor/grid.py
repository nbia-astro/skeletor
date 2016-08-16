from .cython.ppic2_wrapper import grid_t


class Grid(grid_t):

    def __init__(self, nx, ny, comm):

        from .cython.ppic2_wrapper import cpdicomp

        # Number of grid points in x- and y-direction
        self.nx = nx
        self.ny = ny

        self.kstrt = comm.rank + 1
        self.nvp = comm.size

        # edges[0:1] = lower:upper boundary of particle partition
        # nyp = number of primary (complete) gridpoints in particle partition
        # noff = lowermost global gridpoint in particle partition
        # nypmx = maximum size of particle partition, including guard cells
        # nypmn = minimum value of nyp
        edges, nyp, noff, nypmx, nypmn = cpdicomp(ny, self.kstrt, self.nvp)

        if self.nvp > ny:
            msg = "Too many processors requested: ny={}, nvp={}"
            raise RuntimeError(msg.format(ny, self.nvp))

        if nypmn < 1:
            msg = "Combination not supported: ny={}, nvp={}"
            raise RuntimeError(msg.format(ny, self.nvp))

        self.edges = edges
        self.nyp = nyp
        self.noff = noff
        self.nypmx = nypmx
        self.nypmn = nypmn

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
