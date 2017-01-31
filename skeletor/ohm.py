class Ohm:

    """Solve Ohm's law"""

    def __init__(self, manifold, charge=1.0, temperature=0.0, eta=0.0):

        # Store the operators here for easy access
        self.gradient = manifold.gradient
        self.log = manifold.log

        # Charge
        self.charge = charge
        # Temperature
        self.temperature = temperature
        # Resistivity
        self.eta = eta

    @property
    def alpha(self):
        # Ratio of temperature to charge
        return self.temperature/self.charge

    def __call__(self, rho, E):

        self.gradient(self.log(rho), E)
        E['x'] *= -self.alpha/E.grid.dx
        E['y'] *= -self.alpha/E.grid.dy


