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

    def move(self, grid):
        """Uses ppic2's cppmove2 routine for moving particles
           between processors."""

        from .cython.ppic2_wrapper import cppmove2

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

    def periodic_x(self, grid):
        """Applies periodic boundaries on particles along x"""
        from .cython.particle_boundary import periodic_x

        periodic_x(self[:self.np], grid.nx)

    def periodic_y(self, grid):
        """Applies periodic boundaries on particles along y

           Calculates ihole and then calls ppic2's cppmove2 routine for moving
           particles between processors.
        """
        from .cython.particle_boundary import calculate_ihole
        from numpy import array

        calculate_ihole(self[:self.np], self.ihole, array(grid.edges))

        self.move(grid)

    def shear_periodic_y(self, grid, S, t):
        """Shearing periodic boundaries along y.

           Modifies x and vx and applies periodic boundaries
           along y.
        """

        from .cython.particle_boundary import shear_periodic_y

        shear_periodic_y(self[:self.np], grid.ny, S, t)

        self.periodic_y(grid)

    def push_ppic2(self, fxy, dt, t=0):

        from .cython.ppic2_wrapper import cppgbpush2l

        grid = fxy.grid

        # This routine only works for ppic2 grid layout
        assert(grid.nubx == 2 and grid.nuby == 1 and
               grid.lbx == 0 and grid.lby == 0)

        qm = self.charge/self.mass

        ek = cppgbpush2l(self, fxy, self.bz, self.np, self.ihole, qm, dt, grid)

        # Shearing periodicity
        if self.shear:
            self.shear_periodic_y(grid, self.S, t+dt)
        else:
            self.periodic_y(grid)

        # Apply periodicity in x
        self.periodic_x(grid)

        return ek

    def push(self, fxy, dt, t=0):
        from .cython.push_epicycle import push

        grid = fxy.grid
        qtmh = self.charge/self.mass*dt/2
        push(self[:self.np], fxy, self.bz, qtmh, dt, grid.noff,
             grid.lbx, grid.lby)

        # Shearing periodicity
        if self.shear:
            self.shear_periodic_y(grid, self.S, t+dt)
        else:
            self.periodic_y(grid)

        # Apply periodicity in x
        self.periodic_x(grid)
