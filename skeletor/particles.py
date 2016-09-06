import numpy


class Particles(numpy.ndarray):
    """
    Container class for particles in a given subdomain
    """

    def __new__(cls, npmax, charge=1.0, mass=1.0):

        from .cython.dtypes import Int, Particle

        # Size of buffer for passing particles between processors
        nbmax = int(max(0.1*npmax, 1))
        # Size of ihole buffer for particles leaving processor
        ntmax = 2*nbmax

        # Create structured array to hold the particle phase space coordinates
        obj = super().__new__(cls, shape=npmax, dtype=Particle)

        # Particle charge and mass
        obj.charge = charge
        obj.mass = mass

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

        self.charge = getattr(obj, "charge", None)
        self.mass = getattr(obj, "mass", None)
        self.ihole = getattr(obj, "ihole", None)
        self.sbufl = getattr(obj, "sbufl", None)
        self.sbufr = getattr(obj, "sbufr", None)
        self.rbufl = getattr(obj, "rbufl", None)
        self.rbufr = getattr(obj, "rbufr", None)
        self.info = getattr(obj, "info", None)

    def initialize(self, x, y, vx, vy, grid):

        from numpy import logical_and, sum
        from warnings import warn

        ind = logical_and(y >= grid.edges[0], y < grid.edges[1])

        # Number of particles in subdomain
        self.np = sum(ind)

        # Make sure particle array is large enough
        assert self.size >= self.np
        if self.size < int(5/4*self.np):
            msg = "Particle array is probably not large enough"
            warn(msg + " (np={}, npmax={})".format(self.np, self.size))

        # Fill structured array
        self["x"][:self.np] = x[ind]
        self["y"][:self.np] = y[ind]
        self["vx"][:self.np] = vx[ind]
        self["vy"][:self.np] = vy[ind]

        # Probably not where we want this
        self.t = 0.0

    def initialize_ppic2(self, vtx, vty, vdx, vdy, npx, npy, grid):

        from ppic2_wrapper import cpdistr2

        npp, ierr = cpdistr2(self, vtx, vty, vdx, vdy, npx, npy, grid)
        if ierr != 0:
            msg = "Particle initialization error: ierr={}"
            raise RuntimeError(msg.format(ierr))
        self.np = npp

    def periodic_x(self, grid):
        """Periodic boundaries in x"""
        from numpy import where

        ind = numpy.where(self["x"]  < 0)
        self["x"][ind] += grid.Lx
        ind = numpy.where(self["x"] >= grid.Lx)
        self["x"][ind] -= grid.Lx

    def periodic_y(self, grid):
        """Periodic boundaries in y

        This function modifies ihole such that ppic2 can take care of the
        boundary.
        """
        from numpy import where
        from numpy import logical_and

        ih = 0
        nh = 0
        ntmax = self.ihole.shape[0] - 1

        for j in range(self.np):
            if (self["y"][j]<grid.edges[0] or self["y"][j]>=grid.edges[1]):
                if ih < ntmax:
                    self.ihole[ih+1] = j+1
                else:
                    nh = 1
                ih += 1
        if nh > 0:
            ih = -ih
            self.ihole[0] = ih

    def periodic_y_simple(self, grid):
        """Periodic boundaries in y

        This function will not work with MPI.
        """
        from numpy import mod
        self["y"] = mod(self["y"], grid.ny)


    def shear_periodic_x(self, grid):
        """ """
        from numpy import where

        # @TH: These parameters need to be passed. /TB
        # time
        t = self.t
        # Shear parameter
        S = -3/2;

        # Left
        ind1 = numpy.where(self["x"]  < 0)
        # Right
        ind2 = numpy.where(self["x"] >= grid.Lx)

        # Left
        self["x"] [ind1] += grid.Lx
        self["y"] [ind1] += S*grid.Lx*t
        self["vy"][ind1] += S*grid.Lx

        # Right
        self["x"] [ind2] -= grid.Lx
        self["y"] [ind2] -= S*grid.Lx*t
        self["vy"][ind2] -= S*grid.Lx

        # Apply periodicity in y
        self.periodic_y_simple(grid)

    def push(self, fxy, dt, bz=0):

        from .cython.ppic2_wrapper import cppgbpush2l, cppmove2

        grid = fxy.grid

        qm = self.charge/self.mass

        ek = cppgbpush2l(self, fxy, bz, self.np, self.ihole, qm, dt, grid)

        self.shear_periodic_x(grid)

        # Check for ihole overflow error
        if self.ihole[0] < 0:
            ierr = -self.ihole[0]
            msg = "ihole overflow error: ntmax={}, ierr={}"
            raise RuntimeError(msg.format(self.ihole.size - 1, ierr))

        self.np = cppmove2(
                self, self.np, self.sbufl, self.sbufr, self.rbufl,
                self.rbufr, self.ihole, self.info, grid)

        # Increment time
        self.t += dt

        # Make sure particles actually reside in the local subdomain
        assert all(self["y"][:self.np] >= grid.edges[0])
        assert all(self["y"][:self.np] < grid.edges[1])

        return ek
