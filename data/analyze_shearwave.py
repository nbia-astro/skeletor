import numpy as np
from skeletor.reader import read_fields, read_grid
import matplotlib.pyplot as plt
from skeletor.manifolds.second_order import ShearingManifold
from skeletor import Field, Float
from mpi4py.MPI import COMM_WORLD as comm
import matplotlib.animation as animation

# At the moment I have manually recreated the manifold as not enough
# information is stored in the hdf5 files.
nx = 128
ny = 32
Lx = 4
Ly = 1
x0 = -Lx/2
y0 = -Ly/2
S = -3/2
Omega = 1
manifold = ShearingManifold(nx, ny, comm, lbx=2, lby=2,
                            S=S, Omega=Omega, x0=x0, y0=y0, Lx=Lx, Ly=Ly)

# Density field for translating data to primed coordinate system
rho = Field(manifold, dtype=Float)

data_directory = './id0/'
plt.rc('image', aspect='auto', origin='lower')

snaps = 200
t0 = 2*Lx/Ly/S

# Create Fourier amplitude versus time plot
ak = []
time = []
for snap in range(snaps):
    f, h = read_fields(data_directory + 'fields_doubleres_{}.h5'.format(snap))
    rho[:] = f['rho']
    rho.translate(-(t0+h['t']))
    rh = rho.trim().mean(axis=0)
    ak.append(np.fft.rfft(rh)[1])
    time.append(h['t'])

ak = np.array(ak)
time = np.array(time)

plt.figure(2)
plt.clf()
plt.plot(time, ak.real)
plt.xlabel(r'$t$')
plt.ylabel(r'$\mathrm{Re}(\tilde{\rho})$')
plt.savefig('shearing-wave.pdf')


# Create animation of density in the primed coordinate system
snap = 0
f, h = read_fields(data_directory + 'fields_doubleres_{}.h5'.format(snap))

rho[:] = f['rho']
rho.translate(-(t0+h['t']))

# Plot initial condition
plt.figure (1)
plt.clf ()
fig, axes = plt.subplots (num=1, nrows=1)
axes.set_xlabel(r"$x'$")
axes.set_ylabel(r'$\rho$')

ampl = 0.1
axes.set_ylim (1-ampl, 1+ampl)
l1 , = axes.plot(manifold.x, rho.trim().mean(axis=0), 'k')

def iterate(snap):
    f, h = read_fields(data_directory + 'fields_doubleres_{}.h5'.format(snap))

    rho[:] = f['rho']
    rho.translate(-(t0+h['t']))

    l1.set_ydata(rho.trim().mean(axis=0))

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, bitrate=1800)
ani = animation.FuncAnimation(fig, iterate, interval=1, frames=snaps-1)
ani.save('shearing-wave.mp4', writer=writer)