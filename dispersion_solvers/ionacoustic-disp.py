class IonacousticDispersion:
    """Class for solving the ionacoustic dispersion relation"""

    def __init__ (self, Ti, Te, b=1, p=2, dx=1, tol=1e-8,
                  maxterms=1000, N=100):
        from numpy import sqrt, log10, logspace
        # Grid spacing
        self.dx = dx
        # Ion temperature
        self.Ti = Ti
        # Electron temperature
        self.Te = Te
        # Alpha parameter
        self.alpha = sqrt(Ti/Te)
        # Order of interpolation
        self.p = p
        # Energy () or momentum conserving scheme
        self.b = b
        # Tolerance for solutions
        self.tol = tol
        # Max terms in aliasing sum
        self.maxterms = maxterms
        # Number of points used to iterate in temperature
        self.N = N
        # Vector used to iterate alpha up to the target value
        self.alpha_vec = logspace(-4, log10(self.alpha), self.N)

    def W(self, z):
        """Plasma response function"""
        from scipy.special import wofz
        from numpy import pi, sqrt
        return (1. + 1j*sqrt (0.5*pi)*z*wofz (sqrt (0.5)*z))

    def det(self, vph, kx, alpha):
        """Returns the dispersion relation (that needs to be zero)"""
        from numpy import arange, sin, pi
        p = self.p
        b = self.b
        kdx = kx*self.dx
        # The n = 0 term
        A = 1/kdx**(2*p-b+2)*self.W(vph/alpha)

        for j in range(1, self.maxterms):
            B = 0
            for n in (j, -j):
                kn = kdx - 2*pi*n
                B += 1/kn**(2*p-b+2)*self.W(vph/alpha/abs(1-2*pi*n/kdx))
            A += B
            if abs((B/A)) < self.tol:
                A *=  sin(kdx)**(2-b)*(sin(kdx/2)/0.5)**(2*p)
                return A + alpha**2
        raise RuntimeError ("Exceeded maxterms={} aliasing terms!".\
            format (self.maxterms))

    def cold(self, kx):
        """Cold limit of the numerical dispersion relation. Returns the phase
        velocity."""
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

    def omega_vs_alpha(self, kx):
        """Solve warm dispersion relation using Newton's method. Returns the
           complex phase-velocity vph(alpha) where alpha = sqrt(Ti/Te).
           Note that omega = kx*vph."""
        from scipy.optimize import newton
        from numpy import empty, complex128

        det = self.det

        alpha = self.alpha_vec

        # Array for the complex phase-velocity
        vph = empty (len (alpha), dtype = complex128)

        # Initial guess from cold numerical dispersion relation
        vph[0] = self.cold(kx)

        for i in range(1, self.N):
            vph[i] = newton (det, vph[i-1], args = (kx, alpha[i],))
            assert(abs(det(vph[i], kx, alpha[i])) < self.tol), \
            'Guess is not within the tolerance!'

        return vph


    def __call__ (self, kx):
        """Solve warm dispersion relation using Newton's method. Returns the
        complex phase-velocity vph(kx). Note that omega = kx*vph."""
        from scipy.optimize import newton
        from numpy import empty, complex128

        dx    = self.dx
        alpha = self.alpha
        det   = self.det

        # Array for the complex phase-velocity
        vph = empty (len (kx), dtype = complex128)

        # Guess from cold numerical and then iterated up to correct
        # temperature ratio
        vph[0] = self.omega_vs_alpha(kx[0])[-1]

        # Loop over kx
        for i in range (1, len (kx)):
          vph[i] = newton (det, vph[i-1], args = (kx[i], alpha,))
          # Check numerical solution is correct
          assert(abs(det(vph[i], kx[i], alpha)) < self.tol), \
          'Solution is not within the tolerance!'

        return vph

