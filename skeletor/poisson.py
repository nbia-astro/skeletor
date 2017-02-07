class Poisson:

    """Solve Gauss' law ∇·E = ρ/ε0"""

    def __init__(self, manifold, np):

        # Store the operators here for easy access
        self.grad_inv_del = manifold.grad_inv_del

        # Normalization constant
        self.affp = manifold.nx*manifold.ny/np

    def __call__(self, rho, E, **kwds):

        self.grad_inv_del(rho, E, **kwds)
        E['x'] *= self.affp/E.grid.dx
        E['y'] *= self.affp/E.grid.dy
