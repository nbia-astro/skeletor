class Poisson:

    """Solve Gauss' law ∇·E = ρ/ε0"""

    def __init__(self, manifold):

        # Store the operators here for easy access
        self.grad_inv_del = manifold.grad_inv_del

    def __call__(self, rho, E, **kwds):

        self.grad_inv_del(rho, E, **kwds)
