from .cython.types import Float3


class Faraday:

    def __init__(self, manifold):

        from .field import Field

        # Store the curl operator for easy access
        self.curl = manifold.curl

        # Pre-allocate array for dB
        self.dB = Field(manifold, dtype=Float3)

    def __call__(self, E, B, dt, set_boundaries=False):

        # Calculate the curl of E
        self.curl(E, self.dB, down=False)

        # Update the magnetic field
        B['x'] -= self.dB['x']*dt
        B['y'] -= self.dB['y']*dt
        B['z'] -= self.dB['z']*dt

        B.boundaries_set = False

        # Set boundary condition on B?
        if set_boundaries:
            B.copy_guards()
