from skeletor import Float3, Sources, Faraday, Ohm, Field
from skeletor.manifolds.second_order import Manifold
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

eta = 0.01
Lx = 1.
ampl = 0.1

nx, ny = 32, 32
manifold = Manifold(nx, ny, comm, Lx=Lx)

kx = 2*np.pi/Lx

# x- and y-grid
xg, yg = np.meshgrid(manifold.x, manifold.y)

faraday = Faraday(manifold)
ohm = Ohm(manifold, eta=eta)

sources = Sources(manifold)
sources.fill((0.0, 0.0, 0.0, 0.0))
sources.rho.fill(1.0)
sources.copy_guards()

# Set the electric field to zero
E = Field(manifold, dtype=Float3)
E.fill((0.0, 0.0, 0.0))
E.copy_guards()

B = Field(manifold, dtype=Float3)
B.fill((0.0, 0.0, 0.0))
B['y'].active = ampl*np.sin(kx*xg)
B.copy_guards()


def By(t):
    return ampl*np.sin(kx*manifold.x)*np.exp(-eta*kx**2*t)


tend = 10.

dt = 1e-2

nt = int(tend/dt)

if comm.rank == 0:
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, ncols=1)
    im = axes.plot(B['y'].active[0, :], 'C0', By(0), 'C1--')

t = 0
for it in range(nt):
    t += dt
    faraday(E, B, dt, set_boundaries=True)
    ohm(sources, B, E, set_boundaries=True)
    if comm.rank == 0:
        if it % 100:
            im[0].set_ydata(B['y'].active[0, :])
            im[1].set_ydata(By(t))
            plt.pause(1e-7)
