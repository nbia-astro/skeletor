from .cython.types import Float3


class Ohm:

    """Solve Ohm's law"""

    def __init__(self, manifold, charge=1.0, temperature=0.0, eta=0.0):
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

        # Pre-allocate array for electron current and interpolated B-field
        self.Je = Field(manifold, dtype=Float3)
        self.B = Field(manifold, dtype=Float3)
        self.Je.fill((0, 0, 0))
        self.B.fill((0, 0, 0))

    @property
    def alpha(self):
        # Ratio of temperature to charge
        return self.temperature/self.charge

    def __call__(self, sources, B, E, set_boundaries=False):

        # Calculate electron pressure contribution to the electric field
        self.gradient(self.log(sources.rho), E)
        E['x'] *= -self.alpha
        E['y'] *= -self.alpha

        # Ampere's law to calculate total current
        self.curl(B, self.Je)
        # Apply boundary condition to Je
        self.Je.copy_guards()
        # NOTE: This is only done to prevent warnings about doing arithmetic
        # with invalid values in the guard layers. Altenatively (and ideally)
        # we could carry out the computation of Je and the cross product of Je
        # with B only in the active layers.

        for dim in ('x', 'y', 'z'):
            # Subtract ion current to get electron current
            self.Je[dim] -= sources.current[dim]
            # Negative electron fluid velocity
            self.Je[dim] /= sources.rho

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
