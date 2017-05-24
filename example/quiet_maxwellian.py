from skeletor.manifolds.second_order import Manifold
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm
from scipy.special import erfinv
from skeletor.cython.misc import assemple_arrays
import matplotlib.pyplot as plt

# Number of grid points in x- and y-direction
nx, ny = 64, 64
manifold = Manifold(nx, ny, comm, lbx=2, lby=2)

# Number of particles per cell
npc = 2**10

# Thermal velocities in the different directions
vtx = 4
vty = 1
vtz = vty

# Total number of particles in simulation
N = npc*nx*ny
# Number of particles per processor
Np = manifold.nx*manifold.nyp*npc

# Uniform distribution of particle positions (quiet start)
sqrt_npc = int(np.sqrt(npc))
assert sqrt_npc**2 == npc
a = (np.arange(sqrt_npc) + 0.5)/sqrt_npc
x_cell, y_cell = np.meshgrid(a, a)
x_cell = x_cell.flatten()
y_cell = y_cell.flatten()

# Quiet Maxwellian using the inverse error function
R = (np.arange(npc) + 0.5)/npc
vx_cell = erfinv(2*R - 1)*np.sqrt(2)*vtx
vy_cell = erfinv(2*R - 1)*np.sqrt(2)*vty
vz_cell = erfinv(2*R - 1)*np.sqrt(2)*vtz

# Shuffle the velocities to remove some of introduced order
# We use the same shuffling in every cell such that there is still a high
# degree of artifical order in the system.
# In order to ensure this, the seeds have to be the same on every processor.
np.random.seed(1928346143)
np.random.shuffle(vx_cell)
np.random.seed(1928346143+1)
np.random.shuffle(vy_cell)
np.random.seed(1928346143+2)
np.random.shuffle(vz_cell)

# Initialize arrays with particle positions and velocities
# The result has x and y in grid distance units and the velocities in
# 'physical' units.
# Cython is used to bring down the speed of a triple loop.
x = np.empty(Np)
y = np.empty(Np)
vx = np.empty(Np)
vy = np.empty(Np)
vz = np.empty(Np)
assemple_arrays(x_cell, y_cell, vx_cell, vy_cell, vz_cell,
                x, y, vx, vy, vz, npc, manifold)

# Plot positions and velocities
plt.figure(comm.rank)
plt.plot(x, y, '.')
plt.figure(10 + comm.rank)
plt.plot(vx, vy, '.')
plt.show()
