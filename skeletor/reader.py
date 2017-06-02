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


def read_grid(filename):
    from mpi4py.MPI import COMM_WORLD as comm
    from skeletor import Grid

    f = h5py.File(filename, 'r')
    (ny, nx) = f['grid'].attrs['ny_nx']
    (Ly, Lx) = f['grid'].attrs['vsUpperBounds']
    grid = Grid(nx, ny, comm, Lx=Lx, Ly=Ly, lbx=0, lby=0)
    f.close()

    return grid


def read_fields_old(filename):
    from skeletor import Sources, Field, Float3
    """
    Read all the fields from a HDF5 snapshot
    Returns an object with Skeletor fields and sources as attributes.
    """
    grid = read_grid(filename)
    f = h5py.File(filename, 'r')
    E = Field(grid, dtype=Float3)
    E['x'].active = np.array(f['Ex'])
    E['y'].active = np.array(f['Ey'])
    E['z'].active = np.array(f['Ez'])
    B = Field(grid, dtype=Float3)
    B['x'].active = np.array(f['Bx'])
    B['y'].active = np.array(f['By'])
    B['z'].active = np.array(f['Bz'])
    sources = Sources(grid)
    sources.rho.active = np.array(f['rho'])
    sources.Jx.active = np.array(f['rho'])
    sources.Jy.active = np.array(f['rho'])
    sources.Jz.active = np.array(f['rho'])
    t = f['timeGroup'].attrs['vsTime']
    it = f['timeGroup'].attrs['vsStep']
    f.close()
    return Snapshot(E, B, sources, t, it)


def read_fields(filename):
    """
    Read all the fields from a HDF5 snapshot
    """
    Fields = [('Ex', Float), ('Ey', Float), ('Ez', Float),
              ('Bx', Float), ('By', Float), ('Bz', Float),
              ('Jx', Float), ('Jy', Float), ('Jz', Float),
              ('rho', Float)]
    f = h5py.File(filename, 'r')
    shape = np.array(f['Ex']).squeeze().shape
    fields = np.ndarray(shape=shape, dtype=Fields)
    for name in fields.dtype.names:
        fields[name] = np.array(f[name]).squeeze()

    # header information
    t = f['timeGroup'].attrs.__getitem__('vsTime')
    it = f['timeGroup'].attrs.__getitem__('vsStep')
    h = {'t': t, 'it': it}
    f.close()
    return (fields, h)


def read_particles(filename):
    from skeletor import Particle
    f = h5py.File(filename, 'r')
    ions = np.ndarray(f['ions'].shape[0], dtype=Particle)
    ions.view('(5,)f8')[:] = np.array(f['ions'])

    # header information
    t = f['timeGroup'].attrs['vsTime']
    it = f['timeGroup'].attrs['vsStep']
    h = {'t': t, 'it': it}
    f.close()
    return (ions, h)


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
