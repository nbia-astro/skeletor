class IonacousticDispersion:

    def __init__ (self, Ti, Te, b=1, p=2, dx=1, tol=1e-4, maxterms=1000):
        from numpy import sqrt
        self.dx = dx
        self.Ti = Ti
        self.Te = Te
        self.alpha = sqrt(Ti/Te)
        self.p = p
        self.b = b
        self.tol = tol
        self.maxterms = maxterms

    def W(self, z):
        """Plasma response function"""
        from scipy.special import wofz
        from numpy import pi, sqrt
        return (1. + 1j*sqrt (0.5*pi)*z*wofz (sqrt (0.5)*z))

    def det(self, vph, kx):
        """Returns the expression that needs to be zero"""
        from numpy import arange, sin, pi
        # Number of terms included
        p = self.p
        b = self.b
        kdx = kx*self.dx
        # n = 0 term
        A = 1/kdx**(2*p-b+2)*self.W(vph/self.alpha)

        for j in range(1, self.maxterms):
            B = 0
            for n in (j, -j):
                kn = kdx - 2*pi*n
                B += 1/kn**(2*p-b+2)*self.W(vph/self.alpha/abs(1-2*pi*n/kdx))
            A += B
            if abs((B/A)) < self.tol:
                A *=  sin(kdx)**(2-b)*(sin(kdx/2)/0.5)**(2*p)
                return A + self.alpha**2
        raise RuntimeError ("Exceeded maxterms={} aliasing terms!".\
            format (self.maxterms))

    def cold(self, kx):
        """Cold limit of the dispersion relation"""
        from numpy import sin, cos, sqrt
        p = self.p
        b = self.b
        dx= self.dx
        if b == 2:
            if p == 1 or p == 2:
                omega = sqrt((2-2*cos(kx*dx))/dx**2)
            if p == 3:
                omega = sqrt(4*(2 + cos(kx*dx))*sin(kx*dx/2)**2/(3*dx**2))
            if p == 4:
                omega = sqrt((33 + 26*cos(kx*dx) + cos(2*kx*dx))\
                    *sin(kx*dx/2)**2/(15*dx**2))
        if b == 1:
            if p == 1 or p == 2:
                omega = sin(kx*dx)/dx
            if p == 3:
                omega = sqrt((5 + cos(kx*dx))*sin(kx*dx)**2/(6*dx**2))
            if p == 4:
                omega = sqrt((123+56*cos(kx*dx)+cos(2*kx*dx))*sin(kx*dx)**2
                    /(180*dx**2))
        return omega/kx

    def __call__ (self, kx):
        """Solve warm dispersion relation using Newton's method. Returns omega
        which has a real and imaginary part."""
        from scipy.optimize import newton
        from numpy import empty, complex128

        # Array for the complex phase-velocity
        vph = empty (len (kx), dtype = complex128)

        # Initial guess from cold numerical dispersion relation
        guess = self.cold(kx[0])
        # First
        vph[0] = newton (self.det, guess, args = (kx[0]*self.dx,))
        # Loop over kx
        for i in range (1, len (kx)):
          vph[i] = newton (self.det, vph[i-1], args = (kx[i]*self.dx,))
          # Check numerical solution is correct
          assert(abs(self.det(vph[i], kx[i]*self.dx)) < self.tol), \
          'Solution is not with the tolerance!'

        # Frequency of the wave
        omega = kx*vph

        return omega

