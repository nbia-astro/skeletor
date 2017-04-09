from skeletor.cython.types import Float, Float3
from skeletor.field import Field
from skeletor.manifolds.second_order import Manifold
from skeletor.ohm import Ohm
from skeletor.particles import Particles
from skeletor.sources import Sources
from skeletor.state import State
from skeletor.time_steppers.horowitz import TimeStepper
import numpy as np
import matplotlib.pyplot as plt
from mpi4py.MPI import COMM_WORLD as comm
from scipy.special import wofz
from scipy.optimize import newton


def U(z):
    "Plasma response function"
    return -1j*np.sqrt(0.5*np.pi)*wofz(np.sqrt(0.5)*z)


def drel(omega):
    Λp = (omega/(kx*vt))*U((omega + oc)/(kx*vt))
    return kx*kx*va*va + oc*oc*Λp - oc*omega


# Number of grid points in x- and y-direction
nx, ny = 16, 1
# Average number of particles per cell
npc = 64
# Particle charge and mass
charge = 16.0
mass = 1.0
# Dimensionless amplitude of perturbation
ampl = 0.001
# Wavenumber
ikx = 1
# CFL number
cfl = 0.8
# Mean magnetic field strength
B0 = 1.0
# Mean number density
n0 = 1.0
# Plasma beta
β = 20

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
manifold = Manifold(nx, ny, comm)

# Total number of particles in simulation
N = npc*nx*ny

# Alfvén speed
va = B0/np.sqrt(n0*mass)
# Cyclotron frequency
oc = charge*B0/mass
# Thermal velocity
vt = va*np.sqrt(0.5*β)

# Cartesian unit vectors
ex, ey, ez = np.eye(3)

# Wave vector
kx = 2*np.pi*ikx/manifold.Lx
k = kx*ex

# Cold plasma wave frequency
omega = kx*va*(kx*va + np.sqrt(4*oc*oc + kx*kx*va*va))/(2*oc)
# # Warm plasma wave frequency
# omega = newton(drel, omega)
# assert np.isclose(drel(omega), 0.0)

# Grid scale wave number and frequency
kx_m = np.pi/manifold.dx
omega_m = kx_m*va*(kx_m*va + np.sqrt(4*oc*oc + kx_m*kx_m*va*va))/(2*oc)
# Make sure the wave period is resolved on the grid scale
dt = 2**np.floor(np.log2(np.pi/omega_m))
# print('ω = {}'.format(omega_m))
# print(kx_m*va*(kx_m*va - np.sqrt(4*oc*oc + kx_m*kx_m*va*va))/(2*oc))
# print('dt = {}'.format(dt))
# print('oc_grid = {}'.format(va/manifold.dx))

# Circular unit vectors
ep = (ey + 1j*ez)/np.sqrt(2)
em = (ey - 1j*ez)/np.sqrt(2)
assert np.allclose(np.cross(ep, ex), +1j*ep)
assert np.allclose(np.cross(em, ex), -1j*em)

# Lagrangian displacement
xi_hat = ampl*ep
assert np.isclose(np.dot(xi_hat.conj(), xi_hat), ampl*ampl)

# Perturbed magnetic field
B_hat = np.cross(1j*k, np.cross(xi_hat, B0*ex))
# Perturbed current density
J_hat = np.cross(1j*k, B_hat)
# Perturbed electron velocity
ue_hat = -1j*omega*xi_hat
# Perturbed electric field
# E_hat = 1j*omega*np.cross(xi_hat, B0*ex)
E_hat = -np.cross(ue_hat, B0*ex)
# Perturbed ion velocity
u_hat = J_hat/(charge*n0) + ue_hat

def phase(x, y, t):  # noqa: E302
    "Wave phase"
    return np.exp(1j*(kx*x - omega*t))

def By_an(x, y, t):  # noqa: E302
    return (B_hat[1]*phase(x, y, t))
def Bz_an(x, y, t):  # noqa: E302
    return (B_hat[2]*phase(x, y, t))
def uy_an(x, y, t):  # noqa: E302
    return (u_hat[1]*phase(x, y, t))
def uz_an(x, y, t):  # noqa: E302
    return (u_hat[2]*phase(x, y, t))
def Ey_an(x, y, t):  # noqa: E302
    return (E_hat[1]*phase(x, y, t))
def Ez_an(x, y, t):  # noqa: E302
    return (E_hat[2]*phase(x, y, t))

# Maximum number of electrons in each partition
Nmax = int(1.5*N/comm.size)  # noqa: E305

# Particle coordinates
sqrt_npc = int(np.sqrt(npc))
assert sqrt_npc**2 == npc, "'npc' must be a square of an integer."
x, y = [xy.flatten().astype(Float) for xy in np.meshgrid(
    manifold.dx*(np.arange(nx*sqrt_npc) + 0.5)/sqrt_npc,
    manifold.dy*(np.arange(ny*sqrt_npc) + 0.5)/sqrt_npc
    )]
# Particle velocities
vx = np.zeros_like(x)
vy = uy_an(x, y, t=-0.5*dt).real
vz = uz_an(x, y, t=-0.5*dt).real

# Create particle array
ions = Particles(manifold, Nmax, charge=charge, mass=mass, n0=n0)
# Initialize particles
ions.initialize(x, y, vx, vy, vz)

# Initialize sources
sources = Sources(manifold)
# Deposit current
sources.deposit(ions, set_boundaries=True)

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

# Plot ion bulk velocity at t=-dt/2
num = 1
plt.figure(num)
plt.clf()
fig, axes = plt.subplots(num=num, nrows=2, sharex=True)
axes[0].plot(xg.squeeze(), sources.Jy.trim()/(charge*n0), 'k')
axes[1].plot(xg.squeeze(), sources.Jz.trim()/(charge*n0), 'k')
axes[0].plot(xg.squeeze(), uy_an(xg, yg, t=-dt/2).squeeze().real, 'r--')
axes[1].plot(xg.squeeze(), uz_an(xg, yg, t=-dt/2).squeeze().real, 'r--')

# Create vector potential
A_an = Field(manifold, dtype=Float3)
A_an['x'].active = 0.0
A_an['y'].active = +(Bz_an(xg, yg, -dt/2)/(1j*kx)).real
A_an['z'].active = -(By_an(xg, yg, -dt/2)/(1j*kx)).real
A_an.copy_guards()

# Set initial magnetic field perturbation using the vector potential
B = Field(manifold, dtype=Float3)
manifold.curl(A_an, B, down=False)
# Add background magnetic field
B['x'].active += B0
B.copy_guards()

# Plot magnetic field at t=-dt/2
num = 2
plt.figure(num)
plt.clf()
fig, axes = plt.subplots(num=num, nrows=2, sharex=True)
axes[0].plot(xg.squeeze(), B['y'].trim(), 'k')
axes[1].plot(xg.squeeze(), B['z'].trim(), 'k')
axes[0].plot(xg.squeeze(),
             By_an(xg + manifold.dx/2,
                   yg + manifold.dy/2, t=-dt/2).squeeze().real, 'r--')
axes[1].plot(xg.squeeze(),
             Bz_an(xg + manifold.dx/2,
                   yg + manifold.dy/2, t=-dt/2).squeeze().real, 'r--')

# Initialize Ohm's law solver
ohm = Ohm(manifold, temperature=0.0, charge=charge, eta=0.0)

# Initialize state
state = State(ions, B)

# Initialize timestepper
e = TimeStepper(state, ohm, manifold)

# Deposit charges and calculate initial electric field
e.prepare(dt)

# Plot magnetic field at t=0
num = 3
plt.figure(num)
plt.clf()
fig, axes = plt.subplots(num=num, nrows=2, sharex=True)
axes[0].plot(xg.squeeze(), B['y'].trim(), 'k')
axes[1].plot(xg.squeeze(), B['z'].trim(), 'k')
axes[0].plot(xg.squeeze(),
             By_an(xg + manifold.dx/2,
                   yg + manifold.dy/2, t=0.0).squeeze().real, 'r--')
axes[1].plot(xg.squeeze(),
             Bz_an(xg + manifold.dx/2,
                   yg + manifold.dy/2, t=0.0).squeeze().real, 'r--')

# Plot electric field at t=0
num = 4
plt.figure(num)
plt.clf()
fig, axes = plt.subplots(num=num, nrows=2, sharex=True)
axes[0].plot(xg.squeeze(), e.E['y'].trim(), 'k')
axes[1].plot(xg.squeeze(), e.E['z'].trim(), 'k')
axes[0].plot(xg.squeeze(), Ey_an(xg, yg, t=0.0).squeeze().real, 'r--')
axes[1].plot(xg.squeeze(), Ez_an(xg, yg, t=0.0).squeeze().real, 'r--')

plt.draw()
plt.show()
