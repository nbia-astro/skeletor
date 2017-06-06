class HallDispersion():
    """
    This script assumes a magnetic field in the z-direction.
    In order to convert the result to a mean magnetic field in
    the x-direction, the result is rotated around the y-direction: x_new =
    z_old, y_new = y_old and z_new = - x_old.
    FIXME: Generalize to arbitrary orientation in space
    """

    def __init__(self, kperp, va, cs, kpar=0, eta=0, etaH=0, etaA=0,
                 along_x=True, theta=0):

        # Perpendicular wavenumber
        self.kperp = kperp
        # Alfvén speed
        self.va = va
        # Sound speed
        self.cs = cs
        # Paralllel wavenumber
        self.kpar = kpar
        # Ohmic resistivity
        self.eta = eta
        # Hall coefficient
        self.etaH = etaH
        # Ambipolar diffusion (not really relevant)
        self.etaA = etaA
        # True if rotation is needed
        self.along_x = along_x

        self.__call__(kpar, kperp, theta)

    def __call__(self, kpar, kperp, theta):
        import numpy as np

        # Main calculation assumes $\vec{B} = B \hat{z}$
        kx = kperp
        kz = kpar
        eta = self.eta
        etaH = self.etaH
        etaA = self.etaA
        cs = self.cs
        va = self.va

        # Define matrix
        A = np.array([[0, -kx, 0, -kz, 0, 0, 0],
                      [-cs**2*kx, 0, 0, 0, kz*va**2, 0, -kx*va**2],
                      [0, 0, 0, 0, 0, kz*va**2, 0],
                      [-cs**2*kz, 0, 0, 0, 0, 0, 0],
                      [0, kz, 0, 0, -1j*eta*kz**2 - 1j*etaA*kz**2,
                       -1j*etaH*kz**2, 1j*eta*kx*kz + 1j*etaA*kx*kz],
                      [0, 0, kz, 0, 1j*etaH*kz**2, -1j*etaA*kz**2 -
                       1j*eta*(kx**2 + kz**2), -1j*etaH*kx*kz],
                      [0, -kx, 0, 0, 1j*eta*kx*kz + 1j*etaA*kx*kz,
                      1j*etaH*kx*kz, -1j*eta*kx**2 - 1j*etaA*kx**2]])

        # Compute eigenvalues and eigenvectors
        # Notice that we calculate eig(-A)!
        (vals, vecs) = np.linalg.eig(-A)

        if self.along_x:
            # Rotate around y-direction: x_new = z_old, y_new = y_old
            # and z_new = - x_old
            vecs2 = np.copy(vecs)
            # vx
            vecs[1, :] = vecs2[3, :]
            # vz
            vecs[3, :] = -vecs2[1, :]
            # bx
            vecs[4, :] = vecs2[6, :]
            # bz
            vecs[6, :] = -vecs2[4, :]

        # Rotate the solution in the xy-plane (for a perturbation vector
        # inclined with theta wrt to the x-axis)
        from numpy import cos, sin
        vecs2 = np.copy(vecs)
        # vx
        vecs[1, :] = vecs2[1, :]*cos(theta) + vecs[2, :]*sin(theta)
        # vy
        vecs[2, :] = -vecs2[1, :]*sin(theta) + vecs[2, :]*cos(theta)
        # bx
        vecs[4, :] = vecs2[4, :]*cos(theta) + vecs[5, :]*sin(theta)
        # by
        vecs[5, :] = -vecs2[4, :]*sin(theta) + vecs[5, :]*cos(theta)

        # Normalize values
        real_positive = np.abs(np.real(vecs))
        for k in range(7):
            # Normalize eigenvectors
            vecs[:, k] /= np.max(real_positive[:, k])

        # Sort solutions by the value of the eigenvalues
        index = np.argsort(np.real(vals))[::-1]

        # Frequencies
        omega = vals[index]

        # Eigenmode amplitudes
        drho = vecs[0, :][index]
        vx = vecs[1, :][index]
        vy = vecs[2, :][index]
        vz = vecs[3, :][index]
        bx = vecs[4, :][index]
        by = vecs[5, :][index]
        bz = vecs[6, :][index]

        self.omega = omega
        self.vec = [{'drho': drho[k], 'vx': vx[k], 'vy': vy[k], 'vz': vz[k],
                    'bx': bx[k], 'by': by[k], 'bz': bz[k]} for k in range(7)]

if __name__ == '__main__':
    import numpy as np
    kx = 2*np.pi
    va = 1
    cs = 1
    oc = 7

    disp = HallDispersion(kperp=kx, va=va, cs=cs, eta=0.)
