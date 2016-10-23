import numpy as np


def ux(a, t):
    return ampl*np.exp(1j*(kx*a - kappa*t))


def uy(a, t):
    return 2*Omega/(1j*kappa)*ux(a, t)


def uxp(a, t):
    return ux(a, t) + S*t*uy(a, t)


def dxp(a, t):
    duxp = uxp(a, t) - uxp(a, 0)
    duy = uy(a, t) - uy(a, 0)
    return (S*duy/kappa + 1j*duxp)/kappa


def rho(a, t):
    return rho0/(1 + 1j*kx*dxp(a, t))


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    nx = 64

    a = np.arange(nx)

    kx = 2*np.pi/nx

    Omega = 1
    S = -3/2*Omega
    kappa = np.sqrt(2*Omega*(2*Omega+S))
    ampl = 0.1
    dt = 0.01/Omega

    # Density profile at t=0
    rho0 = 1
    # Initial time
    tstart = -5/Omega
    # Duration
    time = 10/Omega

    nt = int(time/dt)

    plt.ion()
    plt.figure(1)
    plt.clf()
    fig, ax = plt.subplots(num=1, ncols=1)

    def real_fundamental(x, y):
        """
        This function does two things:
        * Apply periodic boundary conditions to x-a and y
        * Take the real value
        """
        j = 0
        while x[j] < 0.0:
            x[j] += nx
            j += 1
        x = np.roll(x, -j)
        y = np.roll(y, -j)

        j = -1
        while x[j] > nx:
            x[j] -= nx
            j -= 1
        x = np.roll(x, -j-1)
        y = np.roll(y, -j-1)

        return x.real, y.real

    # Plot initial density
    t = tstart
    line, = ax.plot(*real_fundamental(a + dxp(a, t), rho(a, t)))
    title = ax.set_title('t = {:.2f}'.format(t))
    ax.set_xlim(0, nx)
    ax.set_ylim(0.7, 1.3)
    fig.set_tight_layout(True)

    def animate(it):

        t = tstart + (it+1)*dt

        line.set_data(*real_fundamental(a + dxp(a, t), rho(a, t)))
        title.set_text('t = {:.2f}'.format(t))

        return line, title

    ani = animation.FuncAnimation(
            fig, animate, frames=nt, interval=25, repeat=False)

    plt.show()
