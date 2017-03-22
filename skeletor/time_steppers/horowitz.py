class TimeStepper:

    def __init__(self, state, ohm, manifold):

        from skeletor import Float3, Field, Sources, Faraday

        self.state = state

        # Numerical grid with differential operators
        self.manifold = manifold

        # Ions
        self.ions = state.ions

        # Ohm's law
        self.ohm = ohm

        # Faraday's law
        self.faraday = Faraday(manifold)

        # Initialize sources
        self.sources = Sources(manifold)

        # Set the electric field to zero
        self.E = Field(manifold, dtype=Float3)
        self.E.fill((0.0, 0.0, 0.0))
        self.E.copy_guards()

        self.B = state.B

        # Crate extra arrays (Get rid of some of them later)
        self.E2 = Field(manifold, dtype=Float3)
        self.E3 = Field(manifold, dtype=Float3)
        self.B2 = Field(manifold, dtype=Float3)
        self.B3 = Field(manifold, dtype=Float3)
        self.E4 = Field(manifold, dtype=Float3)
        self.E2.copy_guards()
        self.E3.copy_guards()
        self.B2.copy_guards()
        self.B3.copy_guards()
        self.B2[:] = state.B

        self.t = state.t

    def prepare(self, dt, tol=1.48e-8, maxiter=100):

        from mpi4py.MPI import COMM_WORLD as comm

        # Deposit sources
        self.sources.deposit(self.ions, set_boundaries=True)

        # Calculate electric field (Solve Ohm's law)
        self.ohm(self.sources, self.B, self.E, set_boundaries=True)

        # Drift particle positions by a half time step
        self.ions.drift(dt/2)

        # Iterate to find true electric field at time 0
        for it in range(maxiter):

            # Compute electric field at time 1/2
            self.step(dt, update=False)

            # Average to get electric field at time 0
            for dim in ('x', 'y', 'z'):
                self.E3[dim][...] = 0.5*(self.E[dim] + self.E2[dim])

            # Compute difference to previous iteration
            diff = self.calculate_diff(self.E3, self.E)
            if comm.rank == 0:
                print("Difference to previous iteration: {}".format(diff))

            # Update electric field
            self.E[...] = self.E3

            # Return if difference is sufficiently small
            if diff < tol:
                # Evolve magnetic field by a half step to t=t0
                self.faraday(self.E, self.B, dt/2, set_boundaries=True)
                return

        raise RuntimeError("Exceeded maxiter={} iterations!".format(maxiter))

    def step(self, dt, update=False):
        """Step method from predictor-corrector but update is always false"""
        # Copy magnetic and electric field and ions from previous step
        self.B2[:] = self.B
        self.E2[:] = self.E

        # Evolve magnetic field by a half step to n (n+1)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Push particle positions to n+1 (n+2) and kick velocities to n+1/2
        # (n+3/2). Deposit charge and current at n+1/2 (n+3/2) and only update
        # particle positions if update=True
        self.ions.push_and_deposit(self.E2, self.B2, dt, self.sources, False)

        # Evolve magnetic field by a half step to n+1/2 (n+3/2)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Electric field at n+1/2 (n+3/2)
        self.ohm(self.sources, self.B2, self.E2, set_boundaries=True)

    def calculate_diff(self, f, g):
        from numpy import sqrt
        from mpi4py.MPI import COMM_WORLD as comm, SUM
        diff2 = 0.0
        for dim in ('x', 'y', 'z'):
            diff2 += ((f[dim] - g[dim]).trim()**2).mean()
        diff = sqrt(comm.allreduce(diff2, op=SUM)/comm.size)

        return diff

    def iterate(self, dt, tol=1.48e-8, maxiter=12):
        """Update fields and particles using Horowitz method"""

        # Push and deposit the particles, depositing the sources at n+1/2
        self.ions.push_and_deposit(self.E, self.B, dt, self.sources, True)

        # Start iteration by assuming E^(n+1) = E^n
        self.E3[:] = self.E[:]

        for it in range(maxiter):

            self.E4[:] = self.E3

            # Average electric field to estimate it at n + 1/2
            for dim in ('x', 'y', 'z'):
                self.E2[dim][...] = 0.5*(self.E3[dim] + self.E[dim])

            # Estimate magnetic field at n+1
            self.B3[:] = self.B[:]
            self.faraday(self.E2, self.B3, dt, set_boundaries=True)

            # Estimate magnetic field at n+1/2
            for dim in ('x', 'y', 'z'):
                self.B2[dim][...] = 0.5*(self.B3[dim] + self.B[dim])

            # Estimate electric field at n+1
            self.ohm(self.sources, self.B2, self.E2, set_boundaries=True)

            # New estimate for E^(n+1)
            for dim in ('x', 'y', 'z'):
                self.E3[dim][...] = -self.E[dim] + 2.0*self.E2[dim]

            diff = self.calculate_diff(self.E3, self.E4)

            # Update E and B if difference is sufficiently small
            if (diff < tol):
                self.E[:] = self.E3[:]
                self.B[:] = self.B3[:]
                self.t += dt
                self.state.t = self.t
                return

        raise RuntimeError("Exceeded maxiter={} iterations!".format(maxiter))
