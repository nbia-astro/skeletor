from skeletor import cppinit, Float, Float2, Grid, Particles, Sources
from skeletor import ShearField
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Quiet start
quiet = True

# Plotting
plot = True

# Time step
dt = 0.5e-3

# Simulation time
tend = 2*numpy.pi

# Number of time steps
# nt = int(tend/dt)
nt = 1273

# Particle charge and mass
charge = 1
mass = 1

# Keplerian frequency
Omega = 1

# Shear parameter
S = -3/2

# Epicyclic frequency
kappa = numpy.sqrt(2*Omega*(2*Omega+S))

# Amplitude of perturbation
ampl = 0.1

# Number of grid points in x- and y-direction
nx, ny = 32, 64

# Average number of particles per cell
npc = 16

# Wave numbers
kx = 2*numpy.pi/nx
ky0 = 0

# Total number of particles in simulation
np = npc*nx*ny

if quiet:
    # Uniform distribution of particle positions (quiet start)
    sqrt_npc = int(numpy.sqrt(npc))
    assert sqrt_npc**2 == npc
    dx = dy = 1/sqrt_npc
    x, y = numpy.meshgrid(
            numpy.arange(dx/2, nx+dx/2, dx),
            numpy.arange(dy/2, ny+dy/2, dy))
    x = x.flatten()
    y = y.flatten()
else:
    x = nx*numpy.random.uniform(size=np).astype(Float)
    y = ny*numpy.random.uniform(size=np).astype(Float)

# Particle velocity at t = 0
vx = -S*y
vy = numpy.zeros_like(x)

# Add perturbation
vx += ampl*numpy.cos(kx*x + ky0*y)
vy += 2*Omega/kappa*ampl*numpy.sin(kx*x + ky0*y)

# Drift forward by dt/2
x += vx*dt/2
y += vy*dt/2

# Start parallel processing
idproc, nvp = cppinit(comm)

# Create numerical grid. This contains information about the extent of
# the subdomain assigned to each processor.
grid = Grid(nx, ny, comm)

# x- and y-grid
xx, yy = numpy.meshgrid(grid.x, grid.y)

# Maximum number of ions in each partition
# Set to big number to make sure particles can move between grids
npmax = int(1.25*np/nvp)

# Create particle array
ions = Particles(npmax, charge, mass, Omega=Omega, S=S)

# Assign particles to subdomains
ions.initialize(x, y, vx, vy, grid)

# Make sure particles actually reside in the local subdomain
assert all(ions["y"][:ions.np] >= grid.edges[0])
assert all(ions["y"][:ions.np] < grid.edges[1])

# Make sure the numbers of particles in each subdomain add up to the
# total number of particles
assert comm.allreduce(ions.np, op=MPI.SUM) == np

# Initialize sources
sources = Sources(grid, comm, dtype=Float)
sources.rho = ShearField(grid, comm, dtype=Float)
rho_periodic = ShearField(grid, comm, dtype=Float)

# Deposit sources
sources.deposit(ions)
# assert numpy.isclose(sources.rho.sum(), ions.np*charge)
sources.rho.add_guards(St=0)
# assert numpy.isclose(comm.allreduce(
    # sources.rho.trim().sum(), op=MPI.SUM), np*charge)
sources.rho.copy_guards(0)


def theta(a, t, phi=0):
    return kx*a - kappa*t + phi


def ux(a, t):
    return ampl*numpy.exp(1j*theta(a, t))


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
    return numpy.real(1/alpha(a, t))

a = grid.x

# Electric field in y-direction
E_star = ShearField(grid, comm, dtype=Float2)
E_star.fill((0.0, 0.0))

for i in range(nx+2):
    E_star['y'][:, i] = -2*S*grid.yg*mass/charge*Omega


def concatenate(arr):
    """Concatenate local arrays to obtain global arrays
    The result is available on all processors."""
    return numpy.concatenate(comm.allgather(arr))


# Make initial figure

global_rho = concatenate(sources.rho[:grid.nyp+1, :nx+1])

if comm.rank == 0:
    plt.rc('image', origin='lower', interpolation='nearest', cmap='coolwarm')
    plt.figure(1)
    plt.clf()
    # fig, axes = plt.subplots(num=1, ncols=2)
    # im1 = axes[0].imshow(global_rho)
    # im2 = axes[1].imshow(global_rho)
    # title = axes[0].set_title('it = {}'.format(0))
    fig, axes = plt.subplots(num=1)
    lines = axes.plot(sources.rho[-1, :-1], 'b', sources.rho[-2, :-1], 'g',
                      sources.rho[0, :-1], 'r', sources.rho[1, :-1], 'k')
    axes.set_ylim(6, 17)
    title = axes.set_title('it = {}'.format(0))

t = dt/2

##########################################################################
# Main loop over time                                                    #
##########################################################################


# for it in range(nt):
def animate(it):
    global t
    # Deposit sources
    sources.deposit(ions)
    # assert numpy.isclose(sources.rho.sum(), ions.np*charge)
    # sources.rho.add_guards(S*t)
    # assert numpy.isclose(comm.allreduce(
    #     sources.rho.trim().sum(), op=MPI.SUM), np*charge)

    # sources.rho.copy_guards(S*t)
    # Push particles on each processor. This call also sends and
    # receives particles to and from other processors/subdomains.
    # ions.push(E_star, dt, t=t)
    ions.push_epicycle(E_star, dt, t=t+dt)

    assert comm.allreduce(ions.np, op=MPI.SUM) == np

    # Update time
    t += dt

    # Copy density into a shear field
    rho_periodic[:grid.nyp, :nx] = sources.rho.trim()

    # Translate the density to be periodic in y
    rho_periodic.translate(-S*t)
    rho_periodic.copy_guards(0)

    # Make figures
    if comm.rank == 0:
        # im1.set_data(sources.rho[:grid.nyp+1, :nx+1])
        # im2.set_data(rho_periodic[:grid.nyp+1, :nx+1])
        # im1.autoscale()
        # im2.autoscale()
        lines[0].set_ydata(sources.rho[-1, :-1])
        lines[1].set_ydata(sources.rho[-2, :-1])
        lines[2].set_ydata(sources.rho[0, :-1])
        lines[3].set_ydata(sources.rho[1, :-1])
        title.set_text('it = {}'.format(it))

anim = animation.FuncAnimation(
        fig, animate, frames=nt, interval=25, repeat=False)

# anim.save(
#         'animation.mp4', writer='ffmpeg',
#         fps=25, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])

plt.show()

# Add along x
sources.rho[-1, 0] += sources.rho[-1, -2]
sources.rho[0, 0] += sources.rho[0, -2]
sources.rho[1, 0] += sources.rho[1, -2]
sources.rho[-2, 0] += sources.rho[1, -2]


def translate(g1, trans):
    g1[-1] = g1[0]
    trans %= grid.Ly
    grid.dx = grid.Lx/grid.nx

    # Distance in unit of the grid spacing
    ntrans = trans/grid.dx

    # Integer part
    itrans = numpy.int(numpy.floor(ntrans))

    # Fractional part
    ftrans = ntrans - itrans

    # Fractional shift
    g2 = numpy.empty_like (g1)
    g2[1:] = ftrans*g1[0:-1] + (1 - ftrans)*g1[1:]

    # Boundary
    g2[0] = g2[-1]
    # Integer shift
    return numpy.roll(g2[0:nx], itrans)

t = nt*dt + dt/2

plt.figure(2)
plt.plot(sources.rho[0, :nx])
plt.plot(sources.rho[-1, :nx])

from numpy.fft import rfft, irfft, rfftfreq
kx = 2*numpy.pi*rfftfreq(grid.nx)*grid.Lx/grid.nx
trans = S*grid.Ly*t
shifted = irfft(numpy.exp(-1j*kx*trans)*rfft(sources.rho[-1, :nx]))
shifted2 = translate(sources.rho[-1, :nx+1], trans)
plt.figure(3)
plt.plot(sources.rho[-1, :nx], label='guard')
# plt.plot(shifted, label='Shifted guard - fft')
plt.plot(shifted2, label='Shifted guard - 1st order')
plt.plot(sources.rho[0, :nx], label='Active layer')
plt.legend()

plt.figure(4)
plt.plot(shifted2+sources.rho[0, :nx], label='Sum')
plt.plot(sources.rho[1, :nx], label='Active layer')
plt.legend()
plt.show()
