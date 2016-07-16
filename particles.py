import numpy
from mpi4py.MPI import COMM_WORLD as comm


class Particles(numpy.ndarray):
    """
    Container class for particles in a given subdomain
    """

    # Number of partition boundaries
    idps = 2
    # number of particle phase space coordinates
    idimp = 4
    # ipbc = particle boundary condition: 1 = periodic
    ipbc = 1

    def __new__(cls, x, y, vx, vy, npmax):

        from dtypes import Particle
        from warnings import warn

        # Number of particles in subdomain
        np = x.size

        # Make sure all phase space coordinate arrays have the same size
        assert y.size == vx.size == vy.size == np

        # Make sure particle array is large enough
        assert npmax >= np
        if npmax < int(5/4*np):
            msg = "Particle array is probably not large enough"
            warn(msg + " (np={}, npmax={})".format(np, npmax))

        # Size of buffer for passing particles between processors
        nbmax = int(0.1*npmax)
        # Size of ihole buffer for particles leaving processor
        ntmax = 2*nbmax

        # Create structured array to hold the particle phase space coordinates
        obj = super().__new__(cls, shape=npmax, dtype=Particle)

        # Add additional attributes (see also __array_finalize__)
        obj.np = np
        obj.npmax = npmax
        obj.nbmax = nbmax
        obj.ntmax = ntmax

        # Fill structured array
        obj["x"][:np] = x
        obj["y"][:np] = y
        obj["vx"][:np] = vx
        obj["vy"][:np] = vy

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.np = obj.np
        self.npmax = obj.npmax
        self.nbmax = obj.nbmax
        self.ntmax = obj.ntmax

    def push(self, grid, dt):

        from ppic2_wrapper import cppgpush2l, cppmove2

        ihole, ek = cppgpush2l(
                self, grid.edges, self.np, grid.noff, dt, grid.nx, grid.ny,
                self.idimp, self.npmax, grid.mx, grid.nypmx, self.idps,
                self.ntmax, self.ipbc)

        # Check for ihole overflow error
        if ihole[0] < 0:
            ierr = -ihole[0]
            msg = "ihole overflow error: ntmax={}, ierr={}"
            raise RuntimeError(msg.format(self.ntmax, ierr))

        kstrt = comm.rank + 1
        nvp = comm.size
        self.np, info = cppmove2(
                self, grid.edges, self.np, ihole, grid.ny, kstrt, nvp,
                self.idimp, self.npmax, self.idps, self.nbmax, self.ntmax)

        # Make sure particles actually reside in the local subdomain
        assert all(self["y"][:self.np] >= grid.edges[0])
        assert all(self["y"][:self.np] < grid.edges[1])
