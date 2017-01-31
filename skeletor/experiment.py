class Experiment:

    def __init__ (self, manifold, ions, solver, io=None):

        from skeletor import Float, Float2, Grid, Field, Sources
        from mpi4py.MPI import COMM_WORLD as comm

        # Numerical grid
        self.manifold = manifold

        # Ions
        self.ions = ions

        # Ohm's or Gauss' law
        self.solver = solver

        # IO
        self.io = io

        # Initialize sources
        self.sources = Sources(manifold)

        # Set the electric field to zero
        self.E = Field(manifold, comm, dtype=Float2)
        self.E.fill((0.0, 0.0))

        # Initial time
        self.t = 0.0

    def prepare(self):
        # Deposit sources
        self.sources.deposit(self.ions)

        # Add guards
        self.sources.rho.add_guards()
        self.sources.rho.copy_guards()

        # Calculate electric field (Solve Ohm's law)
        self.solver(self.sources.rho, self.E)
        # Set boundary condition
        self.E.copy_guards()

    def step(self, dt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        self.ions.push(self.E, dt)

        # Update time
        self.t += dt

        # Deposit sources
        self.sources.deposit(self.ions)

        # Boundary calls
        self.sources.rho.add_guards()
        self.sources.rho.copy_guards()

        # Calculate forces (Solve Ohm's or Gauss' law)
        self.solver(self.sources.rho, self.E)

        # Set boundary condition on E
        self.E.copy_guards()


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
