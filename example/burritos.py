import matplotlib.pyplot as plt
import matplotlib.widgets as mw
import numpy as np
import sympy
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
nx = 256

# Lagrangian coordinate
a = sympy.symbols("a")
# Velocity profile in Lagrangian coordinates
# Modify this if desired
k = 2*np.pi/nx
U = 0.1*sympy.sin(k*a)
# Zeroth and first derivative
# These functions accept and return Numpy arrays
U0 = sympy.lambdify(a, U, "numpy")
U1 = sympy.lambdify(a, sympy.diff(U, a), "numpy")

# Eulerian grid coordinate
xg = np.arange(nx)
# Lagrangian particle coordinate
ap = (np.arange(nx*npc) + 0.5)/npc


def update(t):
    """
    This computes the number density both as a function of the *Lagrangian*
    coordinate (or particle label) 'ap' and of the *Eulerian* grid coordinate
    'xg'.
    """
    # Particle velocity
    vp = U0(ap)
    # Eulerian particle coordinate
    xp = ap + vp*t
    # Apply particle boundary condition
    xp %= nx
    # Number density as a function of the particle label
    rhop = 1/(1 + U1(ap)*t)

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
axis.set_ylim(0, 4)
axis.set_xlabel(r'$x$')
axis.set_title(r'$\rho/\rho_0$')

# Create slider widget for changing time
axtime = plt.axes([0.125, 0.1, 0.775, 0.03])
stime = mw.Slider(axtime, 'Time', -300, 300, 0)
stime.on_changed(update)

# Plot number density at t=0
lines = axis.plot(ap, 0*ap, 'k', xg, 0*xg, 'r--')
update(0)

# Update plot as the time is changed interactively
plt.show()
