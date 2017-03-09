class Experiment:

    def __init__ (self, manifold, ions, ohm, B, npc, io=None):

        from skeletor import Float3, Field, Sources, Faraday, Particles
        from mpi4py.MPI import COMM_WORLD as comm

        # Numerical grid with differential operators
        self.manifold = manifold

        # Ions
        self.ions = ions

        # Ohm's law
        self.ohm = ohm

        # Faraday's law
        self.faraday = Faraday(manifold)

        # IO
        self.io = io

        # Initialize sources
        self.sources = Sources(manifold, npc)

        # Set the electric field to zero
        self.E = Field(manifold, comm, dtype=Float3)
        self.E.fill((0.0, 0.0, 0.0))
        self.E.copy_guards()

        self.B = B

        # Crate extra arrays (Get rid of some of them later)
        self.E2 = Field(manifold, comm, dtype=Float3)
        self.E3 = Field(manifold, comm, dtype=Float3)
        self.B2 = Field(manifold, comm, dtype=Float3)
        self.E2.copy_guards()
        self.B2[:] = B

        # Initial time
        self.t = 0.0

    def prepare(self, dt, tol=1.48e-8, maxiter=100):

        from numpy import sqrt
        from mpi4py.MPI import COMM_WORLD as comm, SUM

        # Deposit sources
        self.sources.deposit(self.ions, set_boundaries=True)

        # Calculate electric field (Solve Ohm's law)
        self.ohm(self.sources, self.B, self.E, set_boundaries=True)

        # Drift particle positions by a half time step
        self.ions.drift(dt/2)

        # Iterate to find true electric field at time 0
        for it in range (maxiter):

            # Compute electric field at time 1/2
            self.step (dt, update=False)

            # Average to get electric field at time 0
            for dim in ('x', 'y', 'z'):
                self.E3[dim][...] = 0.5*(self.E[dim] + self.E2[dim])

            # Compute difference to previous iteration
            diff2 = 0.0
            for dim in ('x', 'y', 'z'):
                diff2 += ((self.E3[dim] - self.E[dim]).trim ()**2).mean()
            diff = sqrt(comm.allreduce(diff2, op=SUM)/comm.size)
            if comm.rank == 0:
                print ("Difference to previous iteration: {}".format (diff))

            # Update electric field
            self.E[...] = self.E3

            # Return if difference is sufficiently small
            if diff < tol: return

        raise RuntimeError ("Exceeded maxiter={} iterations!".format (maxiter))

    def step(self, dt, update):

        # Copy magnetic and electric field and ions from previous step
        self.B2[:] = self.B
        self.E2[:] = self.E

        # Evolve magnetic field by a half step to n (n+1)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Push particle positions to n+1 (n+2) and kick velocities to n+1/2
        # (n+3/2). Deposit charge and current at n+1/2 (n+3/2) and only update
        # particle positions if update=True
        self.ions.push_and_deposit(self.E2, self.B2, dt, self.sources, update)

        # Evolve magnetic field by a half step to n+1/2 (n+3/2)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Electric field at n+1/2 (n+3/2)
        self.ohm(self.sources, self.B2, self.E2, set_boundaries=True)

        if update:
            self.B[:] = self.B2
            self.E[:] = self.E2
            self.t += dt

    def iterate(self, dt):

        # Predictor step
        # Get electric field at n+1/2
        self.step (dt, update=True)
        self.E3[...] = self.E2

        # Predict electric field at n+1
        for dim in ('x', 'y', 'z'):
            self.E[dim][...] = 2.0*self.E3[dim] - self.E[dim]

        # Corrector step
        # Get electric field at n+3/2
        self.step (dt, update=False)

        # Predict electric field at n+1
        for dim in ('x', 'y', 'z'):
            self.E[dim][...] = 0.5*(self.E3[dim] + self.E2[dim])

    def run(self, dt, nt):

        if self.io is None:
            for it in range(nt):
                # Update experiment
                self.iterate(dt)
        else:
            # Dump initial data
            self.io.output_fields(self.sources, self.E, self.manifold, self.t)

            for it in range(nt):
                # Update experiment
                self.step(dt)

                self.io.log(it, self.t, dt)

                # Output fields
                if self.io.dt*self.io.snap > self.t:
                    self.io.output_fields(self.sources, self.E, self.manifold,
                                          self.t)
            self.io.finished()
