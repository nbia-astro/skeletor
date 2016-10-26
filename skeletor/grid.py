from .cython.ppic2_wrapper import grid_t


class Grid(grid_t):

    def __init__(self, nx, ny, comm):

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
        edges, nyp, noff, nypmx, nypmn = cpdicomp(ny, comm)

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

    @property
    def yg(self):
        "One-dimensional y-coordinate array including ghost"
        from numpy import arange
        return arange(self.noff, self.noff + self.nyp + 1)


def cpdicomp(ny, comm):
    """Given the number of grid points along y and the MPI communicator
        this function returns grid properties.
    """
    from numpy import zeros
    edges = zeros(2)
    mypm = [0, 0]

    kyp = (ny-1)//comm.size + 1

    edges[0] = kyp*comm.rank

    if edges[0] > ny:
        edges[0] = ny

    noff = edges[0]

    edges[1] = kyp*(comm.rank+1)

    if edges[1] > ny:
        edges[1] = ny

    nyp = edges[1] - noff
    mypm[0] = nyp
    mypm[1] = -nyp
    mypm = comm.allreduce(mypm, op=max)
    nypmx = mypm[0] + 1
    nypmn = -mypm[1]

    if comm.size > ny:
        msg = "Too many processors requested: ny={}, comm.size={}"
        raise RuntimeError(msg.format(ny, comm.size))

    if nypmn < 1:
        msg = "Combination not supported: ny={}, comm.size={}"
        raise RuntimeError(msg.format(ny, comm.size))

    return (edges, int(nyp), int(noff), int(nypmx), int(nypmn))

if __name__ == '__main__':
    # Test if cpdicomp gives same output as ppic2's version
    from mpi4py import MPI
    import numpy as np
    from skeletor import cppinit

    comm = MPI.COMM_WORLD

    # Start parallel processing.
    idproc, nvp = cppinit(comm)

    from skeletor.cython.ppic2_wrapper import cpdicomp as cpdicomp_ppic2
    ny = 256
    kstrt = comm.rank + 1
    nvp = comm.size

    edges, nyp, noff, nypmx, nypmn = cpdicomp_ppic2(ny, kstrt, nvp)
    edges2, nyp2, noff2, nypmx2, nypmn2 = cpdicomp(ny, comm)
    assert(np.array_equal(edges, edges2))
    assert(nyp == nyp2)
    assert(noff == noff2)
    assert(nypmx == nypmx2)
    assert(nypmn == nypmn2)
