import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cbook import mplDeprecation
import warnings
from mpi4py.MPI import COMM_WORLD as comm

plt.rc('image', origin='lower', interpolation='nearest')

# My computer has TkAgg, might be an issue if this is not the case
assert matplotlib.get_backend() == 'TkAgg'

A = np.random.random((25, 100))
A_global = np.concatenate(comm.allgather(A))

if comm.rank == 0:
    fig = plt.figure()
    im = plt.imshow(A_global)

for it in range(10):
    A = np.random.random((25, 100))
    A_global = np.concatenate(comm.allgather(A))
    if comm.rank == 0:
        im.set_array(A_global)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=mplDeprecation)
            plt.pause(0.1)
