from mpi4py import MPI

class Grid:

    # Number of partition boundaries
    idps = 2

    def __init__ (self, nx, ny, nghost=1, comm=MPI.COMM_WORLD):

        from ppic2_wrapper import cpdicomp

        # Number of grid points in x- and y-direction
        self.nx = nx
        self.ny = ny

        # Number of grid points on each side
        self.nghost = nghost

        kstrt = comm.rank + 1
        nvp = comm.size

        # edges[0:1] = lower:upper boundary of particle partition
        # nyp = number of primary (complete) gridpoints in particle partition
        # noff = lowermost global gridpoint in particle partition
        # nypmx = maximum size of particle partition, including guard cells
        # nypmn = minimum value of nyp
        edges, nyp, noff, nypmx, nypmn = cpdicomp (ny, kstrt, nvp, self.idps)

        if nvp > ny:
            msg = "Too many processors requested: ny={}, nvp={}"
            raise RuntimeError (msg.format (ny, nvp))

        if nypmn < 1:
            msg = "Combination not supported: ny={}, nvp={}"
            raise RuntimeError (msg.format (ny, nvp))

        self.edges = edges
        self.nyp = nyp
        self.noff = noff
        self.nypmx = nypmx
        self.nypmn = nypmn

    @property
    def mx (self):
        "Number of total (active plus guard) grid cells in x-direction"
        return self.nx + 2*self.nghost

    @property
    def my (self):
        "Number of total (active plus guard) grid cells in y-direction"
        return self.ny + 2*self.nghost
