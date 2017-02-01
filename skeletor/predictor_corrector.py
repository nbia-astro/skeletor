class Experiment:

    def __init__ (self, manifold, ions, ohm, B, io=None):

        from skeletor import Float2, Field, Sources, Faraday
        from mpi4py.MPI import COMM_WORLD as comm

        # Numerical grid
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

        # Initial time
        self.t = 0.0

    def prepare(self):
        # Deposit sources
        self.sources.deposit(self.ions)

        # Add guards
        self.sources.rho.add_guards()
        self.sources.rho.copy_guards()

        self.sources.J.add_guards_vector()
        self.sources.J.copy_guards()

        # Calculate electric field (Solve Ohm's law)
        self.ohm(self.sources.rho, self.E, self.B, self.sources.J)
        # Set boundary condition
        self.E.copy_guards()

    def step(self, dt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        self.ions.push(self.E, self.B, dt)

        # Update time
        self.t += dt

        # Deposit sources
        self.sources.deposit(self.ions)

        # Boundary calls on sources
        self.sources.rho.add_guards()
        self.sources.rho.copy_guards()
        self.sources.J.add_guards_vector()
        self.sources.J.copy_guards()

        # Ohm's law
        self.ohm(self.sources.rho, self.E, self.B, self.sources.J)
        # Set boundary condition on E
        self.E.copy_guards()

        # Faraday's law
        self.faraday(self.E, self.B, dt)
        # Set boundary condition on B
        self.B.copy_guards()


    def run(self, dt, nt):

        if self.io is None:
            for it in range(nt):
                # Update experiment
                self.step(dt)
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
