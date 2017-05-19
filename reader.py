import h5py
import numpy as np
from skeletor import Float


class Snapshot():

    def __init__(self, E, B, sources, t, it):
        self.E = E
        self.B = B
        self.sources = sources
        self.t = t
        self.it = it


def read_grid(name):
    from mpi4py.MPI import COMM_WORLD as comm
    from skeletor import Grid

    file = h5py.File(name, 'r')
    (ny, nx) = file['grid'].attrs.__getitem__('ny_nx')
    (Ly, Lx) = file['grid'].attrs.__getitem__('vsUpperBounds')
    grid = Grid(nx, ny, comm, Lx=Lx, Ly=Ly, lbx=0, lby=0)
    file.close()

    return grid


def read_fields_old(name):
    from skeletor import Sources, Field, Float3
    """
    Read all the fields from a HDF5 snapshot
    Returns an object with Skeletor fields and sources as attributes.
    """
    grid = read_grid(name)
    file = h5py.File(name, 'r')
    E = Field(grid, dtype=Float3)
    E['x'].active = np.array(file['Ex'])
    E['y'].active = np.array(file['Ey'])
    E['z'].active = np.array(file['Ez'])
    B = Field(grid, dtype=Float3)
    B['x'].active = np.array(file['Bx'])
    B['y'].active = np.array(file['By'])
    B['z'].active = np.array(file['Bz'])
    sources = Sources(grid)
    sources.rho.active = np.array(file['rho'])
    sources.Jx.active = np.array(file['rho'])
    sources.Jy.active = np.array(file['rho'])
    sources.Jz.active = np.array(file['rho'])
    t = file['timeGroup'].attrs.__getitem__('vsTime')
    it = file['timeGroup'].attrs.__getitem__('vsStep')
    file.close()
    f = Snapshot(E, B, sources, t, it)
    return f


def read_fields(name):
    """
    Read all the fields from a HDF5 snapshot
    """
    Fields = [('Ex', Float), ('Ey', Float), ('Ez', Float),
              ('Bx', Float), ('By', Float), ('Bz', Float),
              ('Jx', Float), ('Jy', Float), ('Jz', Float),
              ('rho', Float)]
    file = h5py.File(name, 'r')
    shape = np.array(file['Ex']).squeeze().shape
    f = np.ndarray(shape=shape, dtype=Fields)
    for name in f.dtype.names:
        f[name] = np.array(file[name]).squeeze()

    # header information
    t = file['timeGroup'].attrs.__getitem__('vsTime')
    it = file['timeGroup'].attrs.__getitem__('vsStep')
    h = {'t': t, 'it': it}
    file.close()
    return (f, h)


def read_particles(name):
    # from skeletor import Particle
    Particle = np.dtype(
        [('x', float), ('y', float), ('vx', float), ('vy', float),
         ('vz', float)])
    file = h5py.File(name, 'r')
    ions = np.ndarray(file['ions'].shape[0], dtype=Particle)
    ions.view('(5,)f8')[:] = np.array(file['ions'])
    file.close()
    return ions


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import warnings
    from matplotlib.cbook import mplDeprecation
    plt.rc('image', aspect='auto', interpolation='nearest', origin='lower')

    # Read grid
    grid = read_grid('fields{:04d}.h5'.format(0))
    # Read first snapshot
    f = read_fields_old('fields{:04d}.h5'.format(0))

    # Create figure
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1)
    im = axes.imshow(f.B['y'], extent=(0, grid.Lx, 0, grid.Ly))

    # Plot as a function of time
    for snap in range(1, 90):
        f = read_fields_old('fields{:04d}.h5'.format(snap))
        im.set_data(f.B['y'])
        im.autoscale()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore", category=mplDeprecation)
            plt.pause(1e-7)
