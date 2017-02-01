class Ohm:

    """Solve Ohm's law"""

    def __init__(self, manifold, charge=1.0, temperature=0.0, eta=0.0,
                 npc=None):

        # Store the operators here for easy access
        self.gradient = manifold.gradient
        self.log = manifold.log
        self.curl = manifold.curl

        # Charge
        self.charge = charge
        # Temperature
        self.temperature = temperature
        # Resistivity
        self.eta = eta

        self.npc = npc

    @property
    def alpha(self):
        # Ratio of temperature to charge
        return self.temperature/self.charge

    def __call__(self, sources, B, E, set_boundaries=False):
        from numpy import empty_like

        self.gradient(self.log(sources.rho), E)
        E['x'] *= -self.alpha
        E['y'] *= -self.alpha

        Je = empty_like(sources.J)

        # Ampere's law to calculate total current
        self.curl(B, Je)

        # Subtract ion current to get electron current
        Je['x'] -= sources.J['x']/self.npc
        Je['y'] -= sources.J['y']/self.npc
        Je['z'] -= sources.J['z']/self.npc

        E['x'] += Je['y']*B['z'] - Je['z']*B['y']
        E['y'] += Je['z']*B['x'] - Je['x']*B['z']
        E['z'] += Je['x']*B['y'] - Je['y']*B['x']

        # Convert to dimensionless units
        E['x'] /= E.grid.dx
        E['y'] /= E.grid.dy
        E['z'] /= E.grid.dz

        if set_boundaries:
            E.copy_guards()



