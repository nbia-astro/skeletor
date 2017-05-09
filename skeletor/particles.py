import numpy as np


class Particles(np.ndarray):
    """
    Container class for particles in a given subdomain
    """

    def __new__(cls, manifold, Nmax, time=0.0, charge=1.0, mass=1.0, n0=1.0):

        from .cython.types import Int, Particle

        # Size of buffer for passing particles between processors
        nbmax = int(max(0.1*Nmax, 1))
        # Size of ihole buffer for particles leaving processor
        ntmax = 2*nbmax

        # Create structured array to hold the particle phase space coordinates
        obj = super().__new__(cls, shape=Nmax, dtype=Particle)

        # Particle charge and mass
        obj.charge = charge
        obj.mass = mass
        # Average number density
        obj.n0 = n0

        obj.manifold = manifold

        # Location of hole left in particle arrays
        obj.ihole = np.zeros(ntmax, Int)

        # Buffer arrays for MPI communcation
        obj.sbufl = np.zeros(nbmax, Particle)
        obj.sbufr = np.zeros(nbmax, Particle)
        obj.rbufl = np.zeros(nbmax, Particle)
        obj.rbufr = np.zeros(nbmax, Particle)

        # Info array used for checking errors in particle move
        obj.info = np.zeros(7, Int)

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

        from warnings import warn

        ind = np.logical_and(y >= self.manifold.edges[0]*self.manifold.dy,
                             y < self.manifold.edges[1]*self.manifold.dy)

        # Number of particles in subdomain
        self.N = np.sum(ind)

        # Make sure particle array is large enough
        assert self.size >= self.N
        if self.size < int(5/4*self.N):
            msg = "Particle array is probably not large enough"
            warn(msg + " (N={}, Nmax={})".format(self.N, self.size))

        # Fill structured array
        self["x"][:self.N] = x[ind]/self.manifold.dx
        self["y"][:self.N] = y[ind]/self.manifold.dy
        self["vx"][:self.N] = vx[ind]
        self["vy"][:self.N] = vy[ind]
        self["vz"][:self.N] = vz[ind]

    def move(self):
        """Uses ppic2's cppmove2 routine for moving particles
           between processors."""

        from .cython.ppic2_wrapper import cppmove2

        # Check for ihole overflow error
        if self.ihole[0] < 0:
            ierr = -self.ihole[0]
            msg = "ihole overflow error: ntmax={}, ierr={}"
            raise RuntimeError(msg.format(self.ihole.size - 1, ierr))

        self.N = cppmove2(
                self, self.N, self.sbufl, self.sbufr, self.rbufl,
                self.rbufr, self.ihole, self.info, self.manifold)

        # Make sure particles actually reside in the local subdomain
        assert all(self["y"][:self.N] >= self.manifold.edges[0])
        assert all(self["y"][:self.N] < self.manifold.edges[1])

    def periodic_x(self):
        """Applies periodic boundaries on particles along x"""
        from .cython.particle_boundary import periodic_x

        periodic_x(self[:self.N], self.manifold)

    def periodic_y(self):
        """Applies periodic boundaries on particles along y

           Calculates ihole and then calls ppic2's cppmove2 routine for moving
           particles between processors.
        """
        from .cython.particle_boundary import calculate_ihole

        calculate_ihole(self[:self.N], self.ihole, self.manifold)

        self.move()

    def shear_periodic_y(self):
        """Shearing periodic boundaries along y.

           Modifies x and vx and applies periodic boundaries
           along y.
        """

        from .cython.particle_boundary import shear_periodic_y

        shear_periodic_y(self[:self.N], self.manifold, self.manifold.S,
                         self.time)

        self.periodic_y()

    def push(self, E, B, dt, order='cic'):
        """
        A standard Boris push which updates positions and velocities.

        fxy is the electric field and dt is the time step.
        If shear is turned on, E needs to be E_star and B needs to be B_star
        """
        from .cython.particle_push import boris_push as push

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        if order == 'cic':
            order = 1
        elif order == 'tsc':
            order = 2

        push(self[:self.N], E, B, qtmh, dt, self.manifold, order)

        # Shearing periodicity
        if self.manifold.shear:
            self.shear_periodic_y()
        else:
            self.periodic_y()

        # Apply periodicity in x
        self.periodic_x()

    def push_and_deposit(self, E, B, dt, sources, update=True, order='cic'):
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
        if order == 'cic':
            order = 1
        elif order == 'tsc':
            order = 2
        push_and_deposit(self[:self.N], E, B, qtmh, dt, self.manifold,
                         self.ihole, sources.current, S, update, order)

        # Set boundary flags to False
        sources.current.boundaries_set = False

        # Normalize sources with particle charge
        sources.normalize(self)
        # Add and copy boundary layers
        sources.set_boundaries()

        # Move particles across MPI domains
        if update:
            self.move()

    def push_modified(self, E, B, dt, order='cic'):
        from .cython.particle_push import modified_boris_push as push

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        if order == 'cic':
            order = 1
        elif order == 'tsc':
            order = 2

        push(self[:self.N], E, B, qtmh, dt, self.manifold,
             self.manifold.Omega, self.manifold.S, order)

        # Shearing periodicity
        if self.manifold.shear:
            self.shear_periodic_y()
        else:
            self.periodic_y()

        # Apply periodicity in x
        self.periodic_x()

    def drift(self, dt):
        from .cython.particle_push import drift as cython_drift
        cython_drift(self[:self.N], dt, self.manifold)

        # Apply periodicity in x and y
        self.periodic_x()
        self.periodic_y()
