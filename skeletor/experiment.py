class Experiment:

    def __init__ (self, grid, ions, solver):

        from skeletor import Float, Float2, Grid, Field, Sources
        from mpi4py.MPI import COMM_WORLD as comm

        # Numerical grid
        self.grid = grid

        # Ions
        self.ions = ions

        # Ohm's or Gauss' law
        self.solver = solver

        # Initialize sources
        self.sources = Sources(grid, comm, dtype=Float)

        # Set the electric field to zero
        self.E = Field(grid, comm, dtype=Float2)
        self.E.fill((0.0, 0.0))

        # Initial time
        self.t = 0.0

    def prepare(self):
        # Deposit sources
        self.sources.deposit(self.ions)

        # Add guards
        self.sources.rho.add_guards_ppic2()

        # Calculate electric field (Solve Ohm's law)
        self.solver(self.sources.rho, self.E, destroy_input=False)
        # Set boundary condition
        self.E.copy_guards_ppic2()

    def step(self, dt):
        # Push particles on each processor. This call also sends and
        # receives particles to and from other processors/subdomains.
        self.ions.push(self.E, dt)

        # Update time
        self.t += dt

        # Deposit sources
        self.sources.deposit_ppic2(self.ions)

        # Boundary calls
        self.sources.rho.add_guards_ppic2()

        # Calculate forces (Solve Ohm's or Gauss' law)
        self.solver(self.sources.rho, self.E, destroy_input=False)

        # Set boundary condition on E
        self.E.copy_guards_ppic2()


    def run(self, dt, nt):

        for it in range(nt):
            self.step(dt)
