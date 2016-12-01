class Faraday:

    def __init__(self, operators):
        self.operators = operators

    def __call__(self, E, B, dt):
        from numpy import empty_like
        dB = empty_like(B)

        self.operators.curl(E, dB)

        B['x'] -= dB['x']*dt
        B['y'] -= dB['y']*dt
        B['z'] -= dB['z']*dt