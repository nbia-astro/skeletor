import numpy


class Particles(numpy.ndarray):
    """
    Container class for particles in a given subdomain
    """

    def __new__(cls, npmax, charge=1.0, mass=1.0, S=0, Omega=0, bz=0):

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

        # Constant magnetic field
        obj.bz = bz

        # Shear parameter
        obj.S = S
        # True if shear is turned on
        obj.shear = (S != 0)

        # Angular frequency
        obj.Omega = Omega
        obj.rotation = (Omega != 0)

        if obj.rotation:
            obj.bz += 2.0*mass/charge*Omega

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

    def initialize_ppic2(self, vtx, vty, vdx, vdy, npx, npy, grid):

        from ppic2_wrapper import cpdistr2

        npp, ierr = cpdistr2(self, vtx, vty, vdx, vdy, npx, npy, grid)
        if ierr != 0:
            msg = "Particle initialization error: ierr={}"
            raise RuntimeError(msg.format(ierr))
        self.np = npp

    def periodic_x(self, grid):
        """Periodic boundaries in x

        This function will not work with MPI along x.
        """
        from numpy import mod
        self["x"] = mod(self["x"], grid.nx)

    def shear_periodic_y(self, grid, t):
        """Shearing periodic boundaries along y.

           This function modifies x and vx and subsequently applies periodic
           boundaries on x.

           The periodic boundaries on y are handled by ppic2 *after* we have
           used the values of y to update x and vx.
        """
        from numpy import where

        # Left
        ind1 = where(self["y"] < 0)
        # Right
        ind2 = where(self["y"] >= grid.Ly)

        # Left to right
        self["x"][ind1] -= self.S*grid.Ly*t
        self["vx"][ind1] -= self.S*grid.Ly

        # Right to left
        self["x"][ind2] += self.S*grid.Ly*t
        self["vx"][ind2] += self.S*grid.Ly

    def push(self, fxy, dt, t=0):

        from .cython.ppic2_wrapper import cppgbpush2l, cppmove2

        grid = fxy.grid

        qm = self.charge/self.mass

        ek = cppgbpush2l(self, fxy, self.bz, self.np, self.ihole, qm, dt, grid)

        # Shearing periodicity
        if self.shear:
            self.shear_periodic_y(grid, t+dt)

        # Apply periodicity in x
        self.periodic_x(grid)

        # Check for ihole overflow error
        if self.ihole[0] < 0:
            ierr = -self.ihole[0]
            msg = "ihole overflow error: ntmax={}, ierr={}"
            raise RuntimeError(msg.format(self.ihole.size - 1, ierr))

        self.np = cppmove2(
                self, self.np, self.sbufl, self.sbufr, self.rbufl,
                self.rbufr, self.ihole, self.info, grid)

        # Make sure particles actually reside in the local subdomain
        assert all(self["y"][:self.np] >= grid.edges[0])
        assert all(self["y"][:self.np] < grid.edges[1])

        return ek

    def push_epicycle(self, fxy, dt, t=0):

        from .cython.push_epicycle import push_epicycle as push
        from numpy import mod

        push(self[:self.np], dt)

        # Shearing periodicity
        if self.shear:
            self.shear_periodic_y(fxy.grid, t+dt)

        # Apply periodicity in x
        self["x"] = mod(self["x"], fxy.grid.nx)
        self["y"] = mod(self["y"], fxy.grid.ny)

        return 0.0
