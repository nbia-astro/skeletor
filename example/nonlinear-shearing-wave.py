import numpy as np


def theta(a, t, phi=0):
    return kx*a - kappa*t + phi


def ux(a, t):
    return ampl*np.exp(1j*theta(a, t))


def uy(a, t):
    return 2*Omega/(1j*kappa)*ux(a, t)


def uxp(a, t):
    return ux(a, t) + S*t*uy(a, t)


def xp(a, t):
    A = a - 1/(1j*kappa)*(uxp(a, t) - uxp(a, 0))
    B = S/kappa**2*(uy(a, t) - uy(a, 0))
    return A + B


def alpha(a, t):
    y = 1 + 1j*kx*(xp(a, t) - a)
    return y


def rho(a, t):
    return 1/alpha(a, t)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    nx = 64

    a = np.arange(nx)

    kx = 2*np.pi/nx

    S = -3/2
    Omega = 1
    kappa = np.sqrt(2*Omega*(2*Omega+S))
    ampl = 1e-1

    nt = 500
    t = np.linspace(0, 20*np.pi, nt)

    plt.figure(1)
    plt.clf()
    fig, ax = plt.subplots(num=1, ncols=1)
    im = ax.plot(xp(a, 0), rho(a, 0) - 1)
    ax.set_ylim(-20*ampl, 20*ampl)

    for it in range(nt):
        im[0].set_data(xp(a, t[it]).real, rho(a, t[it]).real - 1)
        plt.pause(1e-2)
