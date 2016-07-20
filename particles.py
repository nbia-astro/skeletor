import numpy


class Particles(numpy.ndarray):
    """
    Container class for particles in a given subdomain
    """

    def __new__(cls, x, y, vx, vy, npmax):

        from dtypes import Int, Particle
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

        # Location of hole left in particle arrays
        obj.ihole = numpy.zeros(ntmax, Int)

        # Buffer arrays for MPI communcation
        obj.sbufl = numpy.zeros(nbmax, Particle)
        obj.sbufr = numpy.zeros(nbmax, Particle)
        obj.rbufl = numpy.zeros(nbmax, Particle)
        obj.rbufr = numpy.zeros(nbmax, Particle)

        # Info array used for checking errors in particle move
        obj.info = numpy.zeros(7, Int)

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

        self.ihole = obj.ihole

    def push(self, fxy, dt):

        from ppic2_wrapper import cppgpush2l, cppmove2

        grid = fxy.grid

        ek = cppgpush2l(
                self, fxy, grid.edges, self.np, grid.noff, self.ihole, dt,
                grid.nx, grid.ny)

        # Check for ihole overflow error
        if self.ihole[0] < 0:
            ierr = -self.ihole[0]
            msg = "ihole overflow error: ntmax={}, ierr={}"
            raise RuntimeError(msg.format(self.ntmax, ierr))

        self.np = cppmove2(
                self, grid.edges, self.np, self.sbufl, self.sbufr, self.rbufl,
                self.rbufr, self.ihole, grid.ny, self.info)

        # Make sure particles actually reside in the local subdomain
        assert all(self["y"][:self.np] >= grid.edges[0])
        assert all(self["y"][:self.np] < grid.edges[1])

        return ek
