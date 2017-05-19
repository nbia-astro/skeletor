import numpy as np

# Example of writing hdf5 files in the VizSchema format
# See https://ice.txcorp.com/trac/vizschema

# Parallel HDF5 can be installed by doing the following
# brew install hdf5 --with-mpi
# brew install h5utils
# The next two lines as one bash command
# CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/usr/local/Cellar/hdf5/1.10.0-patch1
# pip install --no-binary=h5py h5py


def write_grid(file, grid):
    group = file.create_group('grid')
    group.attrs['vsType'] = np.array('mesh', dtype='S')
    group.attrs['vsKind'] = np.array('uniform', dtype='S')
    group.attrs['vsNumCells'] = [grid.ny-1, grid.nx-1]
    group.attrs['ny_nx'] = [grid.ny, grid.nx]
    group.attrs['vsLowerBounds'] = [0., 0.]
    group.attrs['vsUpperBounds'] = [grid.Ly, grid.Lx]


def write_time(file, t, it):
    group = file.create_group('timeGroup')
    group.attrs['vsKind'] = np.array('time', dtype='S')
    group.attrs['vsTime'] = t
    group.attrs['vsStep'] = it


def write_scalar(file, scalar, name):
    phi = file.create_dataset(name, data=scalar)
    phi.attrs['vsType'] = np.array('variable', dtype='S')
    phi.attrs['vsMesh'] = np.array('grid', dtype='S')
    phi.attrs['vsTimeGroup'] = np.array('timeGroup', dtype='S')


def write_fields(file, E=None, B=None, sources=None):
    if E is not None:
        write_scalar(file, E['y'].trim(), 'Ey')
        write_scalar(file, B['z'].trim(), 'Bz')
        write_scalar(file, E['z'].trim(), 'Ez')
    if B is not None:
        write_scalar(file, B['x'].trim(), 'Bx')
        write_scalar(file, E['x'].trim(), 'Ex')
        write_scalar(file, B['y'].trim(), 'By')
    if sources is not None:
        write_scalar(file, sources.rho.trim(), 'rho')
        write_scalar(file, sources.Jx.trim(), 'Jx')
        write_scalar(file, sources.Jy.trim(), 'Jy')
        write_scalar(file, sources.Jz.trim(), 'Jz')


def write_particles(file, ions, N):
    dset = file.create_dataset('ions', (N, 5), dtype=np.float64)
    dset[:, :] = ions.view('(5,)f8')[:N, :]
    dset.attrs['vsType'] = np.array('variableWithMesh', dtype='S')
    dset.attrs['vsNumSpatialDims'] = 2
    dset.attrs['vsLabels'] = np.array('x, y, vx, vy, vz', dtype='S')
