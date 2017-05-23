import numpy as np
from .sources import Sources


class Particles(np.ndarray):
    """
    Container class for particles in a given subdomain
    """

    def __new__(cls, manifold, Nmax,
                time=0.0, charge=1.0, mass=1.0, n0=1.0, order=1):

        from .cython.types import Int, Particle

        msg = 'Interpolation order {} needs more guard layers'.format(order)
        # The number of guard layers on each side needs to be equal to
        # int(ceil(order*0.5 + 0.5)).
        # The additional 0.5 is due to the staggered electric field.
        assert manifold.lbx >= order//2 + 1, msg

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
        # Interpolation order (1 = cic, 2 = tsc)
        obj.order = order

        obj.manifold = manifold

        # Location of hole left in particle arrays
        obj.ihole = np.zeros(ntmax, Int)

        # Buffer arrays for MPI communcation
        obj.sbufl = np.zeros(nbmax, Particle)
        obj.sbufr = np.zeros(nbmax, Particle)
        obj.rbufl = np.zeros(nbmax, Particle)
        obj.rbufr = np.zeros(nbmax, Particle)

        # Create source array
        obj.sources = Sources(manifold)

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
        self.order = getattr(obj, "order", None)
        self.ihole = getattr(obj, "ihole", None)
        self.sbufl = getattr(obj, "sbufl", None)
        self.sbufr = getattr(obj, "sbufr", None)
        self.rbufl = getattr(obj, "rbufl", None)
        self.rbufr = getattr(obj, "rbufr", None)
        self.sources = getattr(obj, "sources", None)
        self.info = getattr(obj, "info", None)
        self.time = getattr(obj, "time", None)

    def initialize(self, x, y, vx, vy, vz):

        from warnings import warn

        # Short hand
        m = self.manifold

        # Array of indices of particles on this processor
        ind = np.logical_and(y >= m.y0 + m.edges[0]*m.dy,
                             y < m.y0 + m.edges[1]*m.dy)

        # Number of particles in subdomain
        self.N = np.sum(ind)

        # Make sure particle array is large enough
        assert self.size >= self.N
        if self.size < int(5/4*self.N):
            msg = "Particle array is probably not large enough"
            warn(msg + " (N={}, Nmax={})".format(self.N, self.size))

        # Fill structured array
        self["x"][:self.N] = (x[ind] - m.x0)/m.dx
        self["y"][:self.N] = (y[ind] - m.y0)/m.dy
        self["vx"][:self.N] = vx[ind]
        self["vy"][:self.N] = vy[ind]
        self["vz"][:self.N] = vz[ind]

    def deposit(self, **kwds):
        self.sources.deposit(self, **kwds)

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

    def push(self, E, B, dt):
        """
        A standard Boris push which updates positions and velocities.

        fxy is the electric field and dt is the time step.
        If shear is turned on, E needs to be E_star and B needs to be B_star
        """
        if self.order == 1:
            from .cython.particle_push import boris_push_cic as push
        elif self.order == 2:
            from .cython.particle_push import boris_push_tsc as push
        else:
            msg = 'Interpolation order {} not implemented.'
            raise RuntimeError(msg.format(self.order))

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        push(self[:self.N], E, B, qtmh, dt, self.manifold)

        # Shearing periodicity
        if hasattr(self.manifold, 'S'):
            self.shear_periodic_y()
        else:
            self.periodic_y()

        # Apply periodicity in x
        self.periodic_x()

    def push_and_deposit(self, E, B, dt, update=True):
        """
        This function updates the particle position and velocities and
        depositis the charge and currents. If update=False only the new
        sources are stored (a predictor step).
        It currently does not work with shear.
        """
        if self.order == 1:
            from .cython.push_and_deposit import push_and_deposit_cic \
                                                    as push_and_deposit
        elif self.order == 2:
            from .cython.push_and_deposit import push_and_deposit_tsc \
                                                    as push_and_deposit
        else:
            msg = 'Interpolation order {} not implemented.'
            raise RuntimeError(msg.format(self.order))

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        # Shear set to zero for the time being
        S = 0.0

        # Zero out the sources
        self.sources.fill((0.0, 0.0, 0.0, 0.0))

        push_and_deposit(self[:self.N], E, B, qtmh, dt, self.manifold,
                         self.ihole, self.sources, S, update)

        # Set boundary flags to False
        self.sources.boundaries_set = False

        # Normalize sources with particle charge
        self.sources.normalize(self)
        # Add and copy boundary layers
        self.sources.set_boundaries()

        # Move particles across MPI domains
        if update:
            self.move()

    def push_modified(self, E, B, dt):
        if self.order == 1:
            from .cython.particle_push import modified_boris_push_cic as push
        elif self.order == 2:
            from .cython.particle_push import modified_boris_push_tsc as push
        else:
            msg = 'Interpolation order {} not implemented.'
            raise RuntimeError(msg.format(self.order))

        # Update time
        self.time += dt

        qtmh = self.charge/self.mass*dt/2

        push(self[:self.N], E, B, qtmh, dt, self.manifold,
             self.manifold.Omega, self.manifold.S)

        # Shearing periodicity
        if hasattr(self.manifold, 'S'):
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
