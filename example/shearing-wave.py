import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def rfft2(R, phase=0.0):
    """
    2D real-to-complex transform in the shearing sheet
    """
    return np.fft.fft(np.exp(-1j*phase)*np.fft.rfft(R), axis=0)


def irfft2(C, phase=0.0):
    """
    Inverse 2D real-to-complex transform in the shearing sheet
    """
    return np.fft.irfft(np.exp(1j*phase)*np.fft.ifft(C, axis=0))


def gradient(psi, t):
    """
    Gradient operator in the shearing sheet
    """
    # We only need to know how much time has elapsed since the last time the
    # domain was strictly periodic
    t %= tS
    # Time dependent part of the laboratory frame 'ky'
    dky = S*t*kx
    # Phase shift to make 'psi' strictly periodic in 'y'.
    # This is an angle, so it can be mapped into the interval [0, 2*pi)
    phase = np.mod(dky*y[:, :nx//2+1], 2*np.pi)
    # Transform real field psi to Fourier space
    psi_k = rfft2(psi, phase)
    # Laboratory frame 'ky'.
    # Exploit periodicity in Fourier space (i.e. aliasing) and make sure that
    #   -π/δy ≤ ky < π/δy
    ky = ky0 + dky
    ky_max = np.pi/dy
    ky = np.mod(ky + ky_max, 2*ky_max) - ky_max
    # Gradient in Fourier space
    ux_k = 1j*kx*psi_k
    uy_k = 1j*ky*psi_k
    # Transform components back to real space
    ux = irfft2(ux_k, phase)
    uy = irfft2(uy_k, phase)

    return ux, uy


def analytic(t):
    """
    ``Analytic solution.'' This consists of a single sheared Fourier harmonic
    in the electrostatic potential Φ and the associated electric field E=-∇Φ
    """
    kx0 = 2*np.pi*ikx/Lx
    phase = kx0*(x + S*t*y)
    Phi = np.cos(phase)
    Ex = kx0*np.sin(phase)
    Ey = S*t*kx0*np.sin(phase)

    return Phi, Ex, Ey


def animate(it):
    """
    Update figure
    """
    # Absolute time
    t = tstart + it*dt
    # Analytic solution
    Phi1, Ex1, Ey1 = analytic(t)
    # Numerically computed electric field
    Ex2, Ey2 = gradient(-Phi1, t)

    image0.set_data(Ex1)
    image1.set_data(Ey1)

    lines01[0].set_xdata(Ex1[j, :])
    lines01[1].set_xdata(Ex2[j, :])

    lines02[0].set_ydata(Ex1[:, i])
    lines02[1].set_ydata(Ex2[:, i])

    lines11[0].set_xdata(Ey1[j, :])
    lines11[1].set_xdata(Ey2[j, :])

    lines12[0].set_ydata(Ey1[:, i])
    lines12[1].set_ydata(Ey2[:, i])

    return image0, image1, lines01, lines02, lines11, lines12


# Domain size
Lx, Ly = 1.0, 1.0
# Number of grid points
nx, ny = 64, 64
# Rate of shear
S = 1.0
# Time step
dt = 2e-2/abs(S)
# Amount of time between instances at which the domain is strictly periodic
tS = Lx/(abs(S)*Ly)
# Start and end time
tstart = -3*tS
tend = 3*tS
# Azimuthal wave number of the ``analytic solution''
ikx = 1

# Total number of time steps
nt = int((tend-tstart)/dt)
# Grid spacing
dx = Lx/nx
dy = Ly/ny
# Grid coordinates
x = (np.arange(nx) - nx/2 + 0.5)*dx
y = (np.arange(ny) - ny/2 + 0.5)*dy
x, y = np.meshgrid(x, y)
# Shearing frame wave numbers
kx = 2*np.pi*np.fft.rfftfreq(nx)/dx
ky0 = 2*np.pi*np.fft.fftfreq(ny)/dy
kx, ky0 = np.meshgrid(kx, ky0)

plt.rc('image', origin='lower', interpolation='nearest', aspect='equal')
plt.rc('axes', labelsize='large')
plt.figure(1)
plt.clf()

fig, axs = plt.subplots(nrows=2, ncols=3, num=1)

for ax in axs.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Get ``analytic solution''
t = tstart
Phi1, Ex1, Ey1 = analytic(t)

# Compute electric field E=-∇Φ numerically
Ex2, Ey2 = gradient(-Phi1, t)

# 2D pseudo-color plots of electric field components
extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
image0 = axs[0, 0].imshow(Ex1, extent=extent)
image1 = axs[1, 0].imshow(Ey1, extent=extent)

# Location of 1D cuts along y and x
i, j = nx//3, 2*ny//3

# Compare analytical(1) and numerical(2) solution for Ex
# along 1D cuts in x and y, respectively
lines01 = axs[0, 1].plot(
        Ex1[j, :], x[j, :], 'k',
        Ex2[j, :], x[j, :], 'r--')
lines02 = axs[0, 2].plot(
        y[:, i], Ex1[:, i], 'k',
        y[:, i], Ex2[:, i], 'r--')

lines11 = axs[1, 1].plot(
        Ey1[j, :], x[j, :], 'k',
        Ey2[j, :], x[j, :], 'r--')
lines12 = axs[1, 2].plot(
        y[:, i], Ey1[:, i], 'k',
        y[:, i], Ey2[:, i], 'r--')

# Visualize the 1D cuts in the 2D pseudo-color plots
for ax in axs[:, 0]:
    ax.axvline(x[j, i], color='k', linewidth=0.5)
    ax.axhline(y[j, i], color='k', linewidth=0.5)

axs[0, 0].set_title('$E_x$')
axs[1, 0].set_title('$E_y$')
axs[1, 0].set_xlabel('$x$')
axs[1, 0].set_ylabel('$y$')
axs[1, 1].set_xlabel('$x$')
axs[1, 2].set_xlabel('$y$')

fig.set_tight_layout(True)

# Start animation
ani = animation.FuncAnimation(fig, animate, np.arange(nt), interval=25)

plt.show()
