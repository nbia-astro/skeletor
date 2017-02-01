class Faraday:

    def __init__(self, operators):
        self.operators = operators

    def __call__(self, E, B, dt, set_boundaries=False):
        from numpy import empty_like
        dB = empty_like(B)

        self.operators.curl(E, dB, down=False)

        B['x'] -= dB['x']*dt
        B['y'] -= dB['y']*dt
        B['z'] -= dB['z']*dt

        B.boundaries_set = False

        if set_boundaries:
            # Set boundary condition on B
            B.copy_guards()