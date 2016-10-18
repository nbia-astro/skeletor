class Ohm:

    """Solve Ohm's law"""

    def __init__(self, grid, npc, charge=1.0, temperature=0.0, eta=0.0):

        # Store the operators here for easy access
        self.operators = grid.operators

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

    def __call__(self, rho, E, destroy_input=True):
        from skeletor import log

        self.operators.gradient(log(rho), E)
        E['x'] *= -self.alpha
        E['y'] *= -self.alpha
