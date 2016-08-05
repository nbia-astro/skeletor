from .cython.ppic2_wrapper import grid_t


class Grid(grid_t):

    def __init__(self, nx, ny, Lx, Ly, comm):

        from .cython.ppic2_wrapper import cpdicomp

        # Number of grid points in x- and y-direction
        self.nx = nx
        self.ny = ny

        self.Lx = Lx
        self.Ly = Ly

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
    def dx(self):
        return self.Lx/self.nx

    @property
    def dy(self):
        return self.Ly/self.ny

