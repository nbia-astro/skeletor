from skeletor import Float3, Field, Particles, Sources
from skeletor import Ohm, DensityPertubation
from skeletor.manifolds.second_order import ShearingManifold
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

plot = True

# Order of particle interpolation
order = 2

# Required number of guard layers on each side
ghost = order//2 + 1

# Quiet start
quiet = True
# Number of grid points in x- and y-direction
nx, ny = 32, 64
# Average number of particles per cell
npc = 256
# Particle charge and mass
charge = 1
mass = 1.0
# Keplerian frequency
Omega = 1

# Shear parameter
S = -3/2

# Epicyclic frequency
kappa = np.sqrt(2*Omega*(2*Omega+S))

# Electron temperature
Te = 1.0
cs = np.sqrt(Te/mass)

# Factor appearing in Ohm's law
alpha = Te/charge

# Box size
Lx, Ly = 2.0, 1.0
x0, y0 = -Lx/2, -Ly/2

# Wavenumbers
ikx = -1
iky = -1

# Wave vector and its modulus
t0 = Lx/Ly/S
kx = 2*np.pi*ikx/Lx
ky0 = 2*np.pi*iky/Ly
k = np.sqrt(kx*kx + ky0*ky0)

kappa2 = 2.0*(2*Omega + S)*Omega

dQ = 0.001
dp_p = -cs*(1j*kx*cs + 2*Omega)/(k**2*cs**2 + kappa2 + 2*S*1j*kx*cs)*dQ
dp_m = -cs*(1j*kx*cs - 2*Omega)/(k**2*cs**2 + kappa2 - 2*S*1j*kx*cs)*dQ
dpy = 0.5*(dp_p + dp_m)
drho = 0.5*(dp_p - dp_m)/cs
dpx  = 1j/ky0*(dQ - 1j*kx*dpy + (2*Omega + S)*drho)
# dpx = -cs*1j*ky0*cs/(k**2 + kappa2)*dQ
rd_hat = drho.real
# CFL number
cfl = 0.2

# Number of periods to run for
nperiods = 1

def ky(t):
    return ky0 + S*t*kx

def phase(x, y, t):
    return np.exp(1j*(kx*x + ky(t)*y))

def rho_an(x, y, t):
    """Analytic density as function of x, y and t"""
    return (1 + drho*phase(x, y, t)).real

def ux_an(x, y, t):
    """Analytic x-velocity as function of x, y and t"""
    return (dpx*phase(x, y, t) - S*y).real/rho_an(x, y, t)

def uy_an(x, y, t):
    """Analytic y-velocity as function of x, y and t"""
    return (dpy*phase(x, y, t)).real/rho_an(x, y, t)

def Ex_an(x, y, t):
    return -alpha*(1j*kx*drho*phase(x, y, t)/rho_an(x, y, t)).real

def Ey_an(x, y, t):
    return -alpha*(1j*ky(t)*drho*phase(x, y, t)/rho_an(x, y, t)).real

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = ShearingManifold(nx, ny, comm, lbx=ghost, lby=ghost,
                            S=S, Omega=Omega, x0=x0, y0=y0, Lx=Lx, Ly=Ly)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Sound speed
cs = np.sqrt(Te/mass)

# Time step
dt = cfl/cs*manifold.dx

# Total number of particles in simulation
N = npc*nx*ny

# Frequency
omega = k*cs

# Simulation time
tend = 5 + t0

# Number of time steps
nt = int(tend/dt)

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)

# Create particle array
ions = Particles(manifold, Nmax, time=t0, charge=charge, mass=mass,
                 order=order)

# Create a uniform density field
init = DensityPertubation(npc, ikx, iky, rd_hat, quiet=quiet)
init(manifold, ions)

# Particle position in physical units
x = x0 + ions['x']*manifold.dx
y = y0 + ions['y']*manifold.dy

# Perturbation to particle velocities
ions['vx'] = ux_an(x, y, t=0)
ions['vy'] = uy_an(x, y, t=0)

# Drift forward by half a time step
# ions.drift(dt/2)
# ions.time += dt/2
# Set boundary condition on particles
ions.shear_periodic_y()
ions.periodic_x()

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.N, op=MPI.SUM) == N

# Set the electric field to zero
E = Field(manifold, dtype=Float3)
E.fill((0.0, 0.0, 0.0))
E.time = t0
E.copy_guards()
B = Field(manifold, dtype=Float3)
B.fill((0.0, 0.0, 0.0))
B.copy_guards()

# Initialize sources
sources = Sources(manifold)

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=Te, charge=charge)

# Calculate initial density and force

# Deposit sources
sources.deposit(ions, set_boundaries=False)
sources.deposit_fix(ions)
sources.add_guards_x()
sources.add_guards_y()
sources.copy_guards()

assert np.isclose(comm.allreduce(
    sources.rho.trim().sum(), op=MPI.SUM), N*charge/npc)

# Calculate electric field (Solve Ohm's law)
ohm(sources, B, E)
# Set boundary condition
E.copy_guards()

# Concatenate local arrays to obtain global arrays
# The result is available on all processors.
def concatenate(arr):
    return np.concatenate(comm.allgather(arr))

# Make initial figure
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.cbook import mplDeprecation
    import warnings
    plt.rc('image', origin='lower', aspect='equal')

    global_rho = concatenate(sources.rho)
    global_Jx = concatenate(sources.Jx)
    global_Jy = concatenate(sources.Jy)
    global_rho_an = concatenate(rho_an(xg, yg, 0))
    global_Jx_an = concatenate((ux_an(xg, yg, 0)+S*yg)*rho_an(xg, yg, 0))
    global_Jy_an = concatenate(uy_an(xg, yg, 0)*rho_an(xg, yg, 0))
    global_Ex_an = concatenate(Ex_an(xg, yg, 0))
    global_Ey_an = concatenate(Ey_an(xg, yg, 0))
    global_E = concatenate(E)
    extent = [x0, x0 + Lx, y0, y0 + Ly]

    if comm.rank == 0:
        plt.rc('image', origin='lower', interpolation='nearest')
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, ncols= 3, nrows=5)
        vmin, vmax = charge*(1 - abs(rd_hat)), charge*(1 + abs(rd_hat))
        im1 = axes[0, 0].imshow(global_rho, vmin=vmin, vmax=vmax,
                                extent=extent)
        im2 = axes[0, 1].imshow(global_rho_an, vmin=vmin, vmax=vmax,
                                extent=extent)
        im3 = axes[1, 0].imshow(global_Jx, extent=extent)
        im4 = axes[1, 1].imshow(global_Jx_an, extent=extent)
        im5 = axes[2, 0].imshow(global_Jy, extent=extent)
        im6 = axes[2, 1].imshow(global_Jy_an, extent=extent)
        im7 = axes[3, 0].imshow(global_E['x'], extent=extent)
        im8 = axes[3, 1].imshow(global_Ex_an, extent=extent)
        im9 = axes[4, 0].imshow(global_E['y'], extent=extent)
        im10 = axes[4, 1].imshow(global_Ey_an, extent=extent)
        # im11 = axes[0, 2].plot(manifold.x, global_rho[ny//2, :], 'b',
        #                       manifold.x, global_rho_an[ny//2, :], 'k--')
        # im12 = axes[1, 2].plot(manifold.x, global_Jx[ny//2, :], 'b',
        #                        manifold.x, global_Jx_an[ny//2, :], 'k--')
        # im13 = axes[2, 2].plot(manifold.x, global_Jy[ny//2, :], 'b',
        #                        manifold.x, global_Jy_an[ny//2, :], 'k--')
        # im14 = axes[3, 2].plot(manifold.x, global_E['x'][ny//2, :], 'b',
        #                        manifold.x, global_Ex_an[ny//2, :], 'k--')
        # im15 = axes[4, 2].plot(manifold.x, global_E['y'][ny//2, :], 'b',
        #                        manifold.x, global_Ey_an[ny//2, :], 'k--')
        # ax1.set_title(r'$\rho$')
        # ax3.set_ylim(vmin, vmax)
        # ax3.set_xlim(x0, x0 + manifold.Lx)

t = t0
diff2 = 0
##########################################################################
# Main loop over time                                                    #
##########################################################################

maxrho = []
minrho = []
rmsrho = []
time = []
for it in range(nt):
    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    ions.order = 1
    ions.push_modified(E, B, dt)

    # Update time
    t += dt

    sources.time = t
    # Deposit sources
    ions.order = 2
    sources.deposit(ions, set_boundaries=False)
    sources.deposit_fix(ions)
    sources.add_guards_x()
    sources.add_guards_y()
    sources.copy_guards()

    E.time = t
    # Calculate forces (Solve Ohm's law)
    ohm(sources, B, E, set_boundaries=True)

    # Difference between numerical and analytic solution
    local_rho = sources.rho.trim()

    maxrho.append(local_rho.max())
    minrho.append(local_rho.min())
    rmsrho.append(np.sqrt(local_rho**2).mean())
    time.append(t)

    local_rho_an = rho_an(xg, yg, t)
    diff2 += ((local_rho_an - local_rho)**2).mean()

    # Make figures
    if plot:
        if (it % 10 == 0):
            print(it, t)
            global_rho = concatenate(sources.rho)
            global_Jx = concatenate(sources.Jx)
            global_Jy = concatenate(sources.Jy)
            global_rho_an = concatenate(rho_an(xg, yg, t))
            global_Jx_an = concatenate((ux_an(xg, yg, t)+S*yg)*rho_an(xg, yg, t))
            global_Jy_an = concatenate(uy_an(xg, yg, t)*rho_an(xg, yg, t))
            global_Ex_an = concatenate(Ex_an(xg, yg, t))
            global_Ey_an = concatenate(Ey_an(xg, yg, t))
            global_E = concatenate(E)
            if comm.rank == 0:
                im1.set_data(global_rho)
                im2.set_data(global_rho_an)
                im3.set_data(global_Jx)
                im4.set_data(global_Jx_an)
                im5.set_data(global_Jy)
                im6.set_data(global_Jy_an)
                im7.set_data(global_E['x'])
                im8.set_data(global_Ex_an)
                im9.set_data(global_E['y'])
                im10.set_data(global_Ey_an)
                for im in (im1, im7, im9):
                    im.autoscale()
                # im11[0].set_ydata(global_rho[ny//2, :])
                # im11[1].set_ydata(global_rho_an[ny//2, :])
                # im12[0].set_ydata(global_Jx[ny//2, :])
                # im12[1].set_ydata(global_Jx_an[ny//2, :])
                # im13[0].set_ydata(global_Jy[ny//2, :])
                # im13[1].set_ydata(global_Jy_an[ny//2, :])
                # im14[0].set_ydata(global_E['x'][ny//2, :])
                # im14[1].set_ydata(global_Ex_an[ny//2, :])
                # im15[0].set_ydata(global_E['y'][ny//2, :])
                # im15[1].set_ydata(global_Ey_an[ny//2, :])
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                            "ignore", category=mplDeprecation)
                    plt.pause(1e-7)