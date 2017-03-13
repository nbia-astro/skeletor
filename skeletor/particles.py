import numpy


class Particles(numpy.ndarray):
    """
    Container class for particles in a given subdomain
    """

    def __new__(cls, manifold, npmax, time=0.0, charge=1.0, mass=1.0, n0=1.0):

        from .cython.types import Int, Particle

        # Size of buffer for passing particles between processors
        nbmax = int(max(0.1*npmax, 1))
        # Size of ihole buffer for particles leaving processor
        ntmax = 2*nbmax

        # Create structured array to hold the particle phase space coordinates
        obj = super().__new__(cls, shape=npmax, dtype=Particle)

        # Particle charge and mass
        obj.charge = charge
        obj.mass = mass
        # Average number density
        obj.n0 = n0

        obj.manifold = manifold

        # Location of hole left in particle arrays
        obj.ihole = numpy.zeros(ntmax, Int)

        # Buffer arrays for MPI communcation
        obj.sbufl = numpy.zeros(nbmax, Particle)
        obj.sbufr = numpy.zeros(nbmax, Particle)
        obj.rbufl = numpy.zeros(nbmax, Particle)
        obj.rbufr = numpy.zeros(nbmax, Particle)

        # Info array used for checking errors in particle move
        obj.info = numpy.zeros(7, Int)

        # Set initial time
        obj.time = time

        return obj

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self.charge = getattr(obj, "charge", None)
        self.mass = getattr(obj, "mass", None)
        self.n0 = getattr(obj, "n0", None)
        self.ihole = getattr(obj, "ihole", None)
        self.sbufl = getattr(obj, "sbufl", None)
        self.sbufr = getattr(obj, "sbufr", None)
        self.rbufl = getattr(obj, "rbufl", None)
        self.rbufr = getattr(obj, "rbufr", None)
        self.info = getattr(obj, "info", None)
        self.time = getattr(obj, "time", None)

    def initialize(self, x, y, vx, vy, vz):

        from numpy import logical_and, sum
        from warnings import warn

        ind = logical_and(y >= self.manifold.edges[0]*self.manifold.dy,
                          y < self.manifold.edges[1]*self.manifold.dy)

        # Number of particles in subdomain
        self.np = sum(ind)

        # Make sure particle array is large enough
        assert self.size >= self.np
        if self.size < int(5/4*self.np):
            msg = "Particle array is probably not large enough"
            warn(msg + " (np={}, npmax={})".format(self.np, self.size))

        # Fill structured array
        self["x"][:self.np] = x[ind]/self.manifold.dx
        self["y"][:self.np] = y[ind]/self.manifold.dy
        self["vx"][:self.np] = vx[ind]
        self["vy"][:self.np] = vy[ind]
        self["vz"][:self.np] = vz[ind]

        self.units = True

    def move(self):
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
                self.rbufr, self.ihole, self.info, self.manifold)

        # Make sure particles actually reside in the local subdomain
        assert all(self["y"][:self.np] >= self.manifold.edges[0])
        assert all(self["y"][:self.np] < self.manifold.edges[1])

    def periodic_x(self):
        """Applies periodic boundaries on particles along x"""
        from .cython.particle_boundary import periodic_x

        periodic_x(self[:self.np], self.manifold.nx)

    def periodic_y(self):
        """Applies periodic boundaries on particles along y

           Calculates ihole and then calls ppic2's cppmove2 routine for moving
           particles between processors.
        """
        from .cython.particle_boundary import calculate_ihole

        calculate_ihole(self[:self.np], self.ihole, self.manifold)

        self.move()

    def shear_periodic_y(self):
        """Shearing periodic boundaries along y.

           Modifies x and vx and applies periodic boundaries
           along y.
        """

        from .cython.particle_boundary import shear_periodic_y

        shear_periodic_y(self[:self.np], self.manifold.ny, self.manifold.S,
                         self.time)

        self.periodic_y()

    def push(self, E, B, dt):
        """
        A standard Boris push which updates positions and velocities.

        fxy is the electric field and dt is the time step.
        If shear is turned on, E needs to be E_star and B needs to be B_star
        """
        from .cython.particle_push import boris_push as push

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        push(self[:self.np], E, B, qtmh, dt, self.manifold)

        # Shearing periodicity
        if self.manifold.shear:
            self.shear_periodic_y()
        else:
            self.periodic_y()

        # Apply periodicity in x
        self.periodic_x()

    def push_and_deposit(self, E, B, dt, sources, update=True):
        """
        This function updates the particle position and velocities and
        depositis the charge and currents. If update=False only the new
        sources are stored (a predictor step).
        It currently does not work with shear.
        """
        from .cython.push_and_deposit import push_and_deposit

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        # Shear set to zero for the time being
        S = 0.0

        # Zero out the sources
        sources.current.fill((0.0, 0.0, 0.0, 0.0))

        push_and_deposit(self[:self.np], E, B, qtmh, dt, self.manifold,
                         self.ihole, sources.current, S, update)

        # Set boundary flags to False
        sources.current.boundaries_set = False

        # Normalize sources with particle charge
        sources.normalize(self)
        # Add and copy boundary layers
        sources.set_boundaries()

        # Move particles across MPI domains
        if update:
            self.move()

    def push_modified(self, E, B, dt):
        from .cython.particle_push import modified_boris_push as push

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        push(self[:self.np], E, B, qtmh, dt, self.manifold,
             self.manifold.Omega, self.manifold.S)

        # Shearing periodicity
        if self.manifold.shear:
            self.shear_periodic_y()
        else:
            self.periodic_y()

        # Apply periodicity in x
        self.periodic_x()

    def drift(self, dt):
        from .cython.particle_push import drift as cython_drift
        cython_drift(self[:self.np], dt, self.manifold)

        # Apply periodicity in x and y
        self.periodic_x()
        self.periodic_y()
