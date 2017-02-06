class Experiment:

    def __init__ (self, manifold, ions, ohm, B, io=None):

        from skeletor import Float2, Field, Sources, Faraday, Particles
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
        self.sources = Sources(manifold)

        # Set the electric field to zero
        self.E = Field(manifold, comm, dtype=Float2)
        self.E.fill((0.0, 0.0, 0.0))

        self.B = B

        # Crate extra arrays (Get rid of some of them later)
        self.ions2 = Particles(manifold, ions.shape[0],
                               charge=ions.charge, mass=ions.mass)
        self.sources2 = Sources(manifold)
        self.E2 = Field(manifold, comm, dtype=Float2)
        self.E3 = Field(manifold, comm, dtype=Float2)
        self.B2 = Field(manifold, comm, dtype=Float2)

        # Initialize some of the
        self.B2[:] = B
        self.ions2[:] = ions
        self.ions2.np = ions.np

        # Initial time
        self.t = 0.0

    def prepare(self, dt, tol=1.48e-8, maxiter=100):

        from numpy import sqrt

        # Deposit sources
        self.sources.deposit(self.ions, set_boundaries=True)

        # Calculate electric field (Solve Ohm's law)
        self.ohm(self.sources, self.B, self.E, set_boundaries=True)

        # Drift particle positions by a half time step
        self.ions['x'] += self.ions['vx']*dt/2
        self.ions['y'] += self.ions['vy']*dt/2

        # Iterate to find true electric field at time 0
        for it in range (maxiter):

            # Compute electric field at time 1/2
            self.step (dt, update=False)

            # Average to get electric field at time 0
            for dim in ('x', 'y', 'z'):
                self.E3[dim][...] = 0.5*(self.E[dim] + self.E2[dim])

            # Compute difference to previous iteration
            diff = 0
            for dim in ('x', 'y', 'z'):
                diff += sqrt(((self.E3[dim] - self.E[dim]).trim ()**2).mean())
            if self.manifold.comm.rank == 0:
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
        self.ions2[:] = self.ions

        # Evolve magnetic field by a half step to n (n+1)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Push particle positions to n+1 (n+2) and
        # velocities to n+1/2 (n+3/2)
        self.ions2.push(self.E2, self.B2, dt)

        # Deposit sources
        self.sources2.deposit(self.ions2, set_boundaries=True)

        # Evolve magnetic field by a half step to n+1/2 (n+3/2)
        self.faraday(self.E2, self.B2, dt/2, set_boundaries=True)

        # Electric field at n+1/2 (n+3/2)
        self.ohm(self.sources2, self.B2, self.E2, set_boundaries=True)

        if update:
            self.B[:] = self.B2
            self.E[:] = self.E2
            self.ions[:] = self.ions2
            self.sources.rho[:] = self.sources2.rho
            self.sources.J[:] = self.sources2.J
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
