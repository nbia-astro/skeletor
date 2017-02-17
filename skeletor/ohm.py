from .cython.types import Float2

class Ohm:

    """Solve Ohm's law"""

    def __init__(self, manifold, charge=1.0, temperature=0.0, eta=0.0,
                 npc=None):
        from .field import Field

        # Store the operators here for easy access
        self.gradient = manifold.gradient
        self.log = manifold.log
        self.curl = manifold.curl
        self.unstagger = manifold.unstagger

        # Charge
        self.charge = charge
        # Temperature
        self.temperature = temperature
        # Resistivity
        self.eta = eta

        # Particles per cell
        self.npc = npc

        # Pre-allocate array for electron current and interpolated B-field
        self.Je = Field(manifold, dtype=Float2)
        self.B = Field(manifold, dtype=Float2)
        self.Je.fill((0,0,0))
        self.B.fill((0,0,0))

    @property
    def alpha(self):
        # Ratio of temperature to charge
        return self.temperature/self.charge

    def __call__(self, sources, B, E, set_boundaries=False):

        # Normalize charge and current with number of particles per cell (npc)
        # This really should be done in sources.py
        sources.rho /= self.npc
        for dim in ('x', 'y', 'z'):
            sources.J[dim] /= self.npc

        # Short-hand for rho
        rho = sources.rho

        # Calculate electron pressure contribution to the electric field
        self.gradient(self.log(rho), E)
        E['x'] *= -self.alpha
        E['y'] *= -self.alpha

        # Ampere's law to calculate total current
        self.curl(B, self.Je)


        for dim in ('x', 'y', 'z'):
            # Subtract ion current to get electron current
            self.Je[dim] -= sources.J[dim]
            # Negative electron fluid velocity
            self.Je[dim] /= rho

        # Interpolate the B-field onto the location of E
        self.unstagger(B, self.B)

        # Take the cross product J_e times B
        E['x'] += self.Je['y']*self.B['z'] - self.Je['z']*self.B['y']
        E['y'] += self.Je['z']*self.B['x'] - self.Je['x']*self.B['z']
        E['z'] += self.Je['x']*self.B['y'] - self.Je['y']*self.B['x']

        E.boundaries_set = False

        # Set boundary condition on E?
        if set_boundaries:
            E.copy_guards()


