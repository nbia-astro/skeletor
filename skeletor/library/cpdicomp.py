
def cpdicomp(ny, comm):
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
    from mpi4py import MPI
    import numpy as np
    from skeletor import cppinit

    comm = MPI.COMM_WORLD

    # Start parallel processing.
    idproc, nvp = cppinit(comm)

    from skeletor.cython.ppic2_wrapper import cpdicomp as cpdicomp_ppic2

    # Number of grid points in x- and y-direction
    ny = 256

    kstrt = comm.rank + 1
    nvp = comm.size

    # edges[0:1] = lower:upper boundary of particle partition
    # nyp = number of primary (complete) gridpoints in particle partition
    # noff = lowermost global gridpoint in particle partition
    # nypmx = maximum size of particle partition, including guard cells
    # nypmn = minimum value of nyp
    edges, nyp, noff, nypmx, nypmn = cpdicomp_ppic2(ny, kstrt, nvp)

    edges2, nyp2, noff2, nypmx2, nypmn2 = cpdicomp(ny, comm)
    print('')
    print('edges', edges)
    print('edges2', edges2)
    assert(np.array_equal(edges, edges2))
    print('nyp', nyp)
    print('nyp2', nyp2)
    assert(nyp == nyp2)
    print('noff', noff)
    print('noff2', noff2)
    assert(noff == noff2)
    print('nypmx', nypmx)
    print('nypmx2', nypmx2)
    assert(nypmx == nypmx2)
    print('nypmn', nypmn)
    print('nypmn2', nypmn2)
    assert(nypmn == nypmn2)