import numpy


class Particles(numpy.ndarray):
    """
    Container class for particles in a given subdomain
    """

    def __new__(cls, npmax):

        from dtypes import Int, Particle

        # Size of buffer for passing particles between processors
        nbmax = int(0.1*npmax)
        # Size of ihole buffer for particles leaving processor
        ntmax = 2*nbmax

        # Create structured array to hold the particle phase space coordinates
        obj = super().__new__(cls, shape=npmax, dtype=Particle)

        # Location of hole left in particle arrays
        obj.ihole = numpy.zeros(ntmax, Int)

        # Buffer arrays for MPI communcation
        obj.sbufl = numpy.zeros(nbmax, Particle)
        obj.sbufr = numpy.zeros(nbmax, Particle)
        obj.rbufl = numpy.zeros(nbmax, Particle)
        obj.rbufr = numpy.zeros(nbmax, Particle)

        # Info array used for checking errors in particle move
        obj.info = numpy.zeros(7, Int)

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.ihole = obj.ihole
        self.sbufl = obj.sbufl
        self.sbufr = obj.sbufr
        self.rbufl = obj.rbufl
        self.rbufr = obj.rbufr
        self.info = obj.info

    def initialize(self, x, y, vx, vy):

        from warnings import warn

        # Number of particles in subdomain
        self.np = x.size

        # Make sure all phase space coordinate arrays have the same size
        assert y.size == vx.size == vy.size == self.np

        # Make sure particle array is large enough
        assert self.size >= self.np
        if self.size < int(5/4*self.np):
            msg = "Particle array is probably not large enough"
            warn(msg + " (np={}, npmax={})".format(self.np, self.size))

        # Fill structured array
        self["x"][:self.np] = x
        self["y"][:self.np] = y
        self["vx"][:self.np] = vx
        self["vy"][:self.np] = vy

    def push(self, fxy, dt):

        from ppic2_wrapper import cppgpush2l, cppmove2

        grid = fxy.grid
        comm = fxy.comm

        ek = cppgpush2l(self, fxy, self.np, self.ihole, dt, grid)

        # Check for ihole overflow error
        if self.ihole[0] < 0:
            ierr = -self.ihole[0]
            msg = "ihole overflow error: ntmax={}, ierr={}"
            raise RuntimeError(msg.format(self.ihole.size - 1, ierr))

        self.np = cppmove2(
                self, self.np, self.sbufl, self.sbufr, self.rbufl,
                self.rbufr, self.ihole, self.info, grid, comm)

        # Make sure particles actually reside in the local subdomain
        assert all(self["y"][:self.np] >= grid.edges[0])
        assert all(self["y"][:self.np] < grid.edges[1])

        return ek
