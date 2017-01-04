import matplotlib.pyplot as plt
import matplotlib.widgets as mw
import numpy as np
import numba


@numba.jit(numba.void(numba.float64[:], numba.float64[:]))
def deposit(pos, den):
    """
    Given particle positions, compute number density using cloud-in-cell
    interpolation.
    """
    # Erase density field
    for j in range(den.shape[0]):
        den[j] = 0.0

    # Scatter particles onto the grid
    for ip in range(pos.shape[0]):

        j = int(pos[ip])
        w = pos[ip] - j

        den[j] += 1.0 - w
        den[j+1] += w

    # Add density from guard cell to corresponding active cell
    den[0] += den[-1]
    # Apply boundary condition
    den[-1] = den[0]


# Number of particles per cell
npc = 1024
# Number of grid points
nx = 64

# Eulerian grid coordinate
xg = np.arange(nx)
# Lagrangian particle coordinate
ap = (np.arange(nx*npc) + 0.5)/npc

bp = 0

# Particle charge and mass
charge = 1
mass = 1

# Keplerian frequency
Omega = 1

# Shear parameter
S = -3/2

# Epicyclic frequency
kappa = np.sqrt(2*Omega*(2*Omega+S))

# Amplitude of perturbation
ampl = 0.5

# Wavenumber
kx = 2*np.pi/nx

# Phase
phi = kx*ap


def x_an(t):
    x = 2*Omega/kappa*ampl*(np.sin(kappa*t + phi) - np.sin(phi)) + ap \
        - S*t*(bp - ampl*np.cos(phi))
    return x


def y_an(t):
    y = ampl*(np.cos(kappa*t + phi) - np.cos(phi)) + bp
    return y


def alpha_particle(a, t):
    dxda = 2*Omega/kappa*ampl*kx*(np.cos(kappa*t + phi) - np.cos(phi)) + 1 \
        - S*t*ampl*kx*np.sin(phi)
    dyda = -ampl*kx*(np.sin(kappa*t + phi) - np.sin(phi))

    return dxda + S*t*dyda


def rho_an_particle(a, t):
    return 1/alpha_particle(a, t)


def update(t):
    """
    This computes the number density both as a function of the *Lagrangian*
    coordinate (or particle label) 'ap' and of the *Eulerian* grid coordinate
    'xg'.
    """
    # Particle velocity
    # vp = U0(ap)
    # Eulerian particle coordinate
    xp = x_an(t) + S*y_an(t)*t
    # Apply particle boundary condition
    xp %= nx
    # Number density as a function of the particle label
    rhop = rho_an_particle(ap, t)

    # Initialize density
    rho = np.empty(nx+1, dtype=np.double)
    # Scatter particles onto the grid
    deposit(xp, rho)
    # # Make sure all particles have been accounted for
    # assert np.isclose(rho[:-1].sum(), nx*npc)

    # Update plot
    lines[0].set_data(xp, rhop)
    lines[1].set_data(xg, rho[:-1]/npc)


# Create figure
plt.figure(1)
plt.clf()
fig, axis = plt.subplots(num=1)
plt.subplots_adjust(bottom=0.25)
axis.set_ylim(0, 2)
axis.set_xlabel(r'$x$')
axis.set_title(r'$\rho/\rho_0$')

# Create slider widget for changing time
axtime = plt.axes([0.125, 0.1, 0.775, 0.03])
stime = mw.Slider(axtime, 'Time', -2*np.pi, 2*np.pi, 0)
stime.on_changed(update)

# Plot number density at t=0
lines = axis.plot(ap, 0*ap, 'r-', xg, 0*xg, 'k--')
plt.legend(lines, ['Lagrangian theory', 'Eulerian deposition'],
           loc='upper left')
update(0)

# Update plot as the time is changed interactively
plt.show()
