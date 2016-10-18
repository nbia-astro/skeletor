class Poisson:

    """Solve Gauss' law ∇·E = ρ/ε0"""

    def __init__(self, grid, ax, ay, np):

        # Store the operators here for easy access
        self.operators = grid.operators

        # Normalization constant
        self.affp = grid.nx*grid.ny/np

    def __call__(self, rho, E, destroy_input=True):

        self.operators.grad_inv_del(rho, E)
        E['x'] *= self.affp
        E['y'] *= self.affp
