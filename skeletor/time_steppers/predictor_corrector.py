class TimeStepper:

    def __init__(self, state, ohm, manifold):

        from skeletor import Float3, Field, Sources, Faraday

        self.state = state

        # Numerical grid with differential operators
        self.manifold = manifold

        # Ohm's law
        self.ohm = ohm

        # Faraday's law
        self.faraday = Faraday(manifold)

        # Initialize sources
        self.sources = Sources(manifold)
        # Used for storing
        self.sources2 = Sources(manifold)

        # Set the electric field to zero
        self.E = Field(manifold, dtype=Float3)
        self.E.fill((0.0, 0.0, 0.0))
        self.E.copy_guards()

        self.B = state.B

        # Crate extra arrays (Get rid of some of them later)
        self.E2 = Field(manifold, dtype=Float3)
        self.E3 = Field(manifold, dtype=Float3)
        self.B2 = Field(manifold, dtype=Float3)
        self.E2.copy_guards()
        self.B2[:] = state.B

        # Initial time
        self.t = state.t

    def prepare(self, dt, tol=1.48e-8, maxiter=100):

        from numpy import sqrt
        from mpi4py.MPI import COMM_WORLD as comm, SUM

        # Deposit sources
        self.sources.current.fill((0.0, 0.0, 0.0, 0.0))
        for ions in self.state.species:
            self.sources2.deposit(ions, set_boundaries=True)
            for dim in self.sources.current.dtype.names:
                self.sources.current[dim] += self.sources2.current[dim]
        self.sources.current.boundaries_set = True

        # Calculate electric field (Solve Ohm's law)
        self.ohm(self.sources, self.B, self.E, set_boundaries=True)

        # Drift particle positions by a half time step
        for ions in self.state.species:
            ions.drift(dt/2)

        # Iterate to find true electric field at time 0
        for it in range(maxiter):

            # Compute electric field at time 1/2
            self.step(dt, update=False)

            # Average to get electric field at time 0
            for dim in ('x', 'y', 'z'):
                self.E3[dim][...] = 0.5*(self.E[dim] + self.E2[dim])

            # Compute difference to previous iteration
            diff2 = 0.0
            for dim in ('x', 'y', 'z'):
                diff2 += ((self.E3[dim] - self.E[dim]).trim()**2).mean()
            diff = sqrt(comm.allreduce(diff2, op=SUM)/comm.size)
            if comm.rank == 0:
                print("Difference to previous iteration: {}".format(diff))

            # Update electric field
            self.E[...] = self.E3

            # Return if difference is sufficiently small
            if diff < tol:
                return

        raise RuntimeError("Exceeded maxiter={} iterations!".format(maxiter))

    def step(self, dt, update):

        # Copy magnetic and electric field and ions from previous step
        self.B2[:] = self.B
        self.E2[:] = self.E

        # Evolve magnetic field by a half step to n (n+1)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Push particle positions to n+1 (n+2) and kick velocities to n+1/2
        # (n+3/2). Deposit charge and current at n+1/2 (n+3/2) and only update
        # particle positions if update=True
        self.sources.current.fill((0.0, 0.0, 0.0, 0.0))
        self.sources.current.boundaries_set = False
        for ions in self.state.species:
            ions.push_and_deposit(self.E2, self.B2, dt, self.sources2, update)
            for dim in self.sources.current.dtype.names:
                self.sources.current[dim] += self.sources2.current[dim]
        self.sources.current.boundaries_set = True

        # Evolve magnetic field by a half step to n+1/2 (n+3/2)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Electric field at n+1/2 (n+3/2)
        self.ohm(self.sources, self.B2, self.E2, set_boundaries=True)

        if update:
            self.B[:] = self.B2
            self.E[:] = self.E2
            self.t += dt
            self.state.t = self.t

    def iterate(self, dt):

        # Predictor step
        # Get electric field at n+1/2
        self.step(dt, update=True)
        self.E3[...] = self.E2

        # Predict electric field at n+1
        for dim in ('x', 'y', 'z'):
            self.E[dim][...] = 2.0*self.E3[dim] - self.E[dim]

        # Corrector step
        # Get electric field at n+3/2
        self.step(dt, update=False)

        # Predict electric field at n+1
        for dim in ('x', 'y', 'z'):
            self.E[dim][...] = 0.5*(self.E3[dim] + self.E2[dim])
