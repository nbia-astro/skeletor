import numpy as np

# Example of writing hdf5 fs in the VizSchema format
# See https://ice.txcorp.com/trac/vizschema

# Parallel HDF5 can be installed by doing the following
# brew install hdf5 --with-mpi
# brew install h5utils
# The next two lines as one bash command
# CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/usr/local/Cellar/hdf5/1.10.0-patch1
# pip install --no-binary=h5py h5py


def write_grid(f, grid):
    group = f.create_group('grid')
    group.attrs['vsType'] = np.array('mesh', dtype='S')
    group.attrs['vsKind'] = np.array('uniform', dtype='S')
    group.attrs['vsStartCell'] = [0, 0]
    group.attrs['vsNumCells'] = [grid.nyp-1, grid.nx-1]
    group.attrs['ny_nx'] = [grid.ny, grid.nx]
    group.attrs['lby_lbx'] = [grid.lby, grid.lbx]
    group.attrs['vsLowerBounds'] = [grid.noff*grid.dy, 0.]
    group.attrs['vsUpperBounds'] = [(grid.noff + grid.nyp)*grid.dy, grid.Lx]


def write_time(f, t, it):
    group = f.create_group('timeGroup')
    group.attrs['vsKind'] = np.array('time', dtype='S')
    group.attrs['vsTime'] = t
    group.attrs['vsStep'] = it


def write_scalar(f, scalar, name):
    phi = f.create_dataset(name, data=scalar)
    phi.attrs['vsType'] = np.array('variable', dtype='S')
    phi.attrs['vsMesh'] = np.array('grid', dtype='S')
    phi.attrs['vsTimeGroup'] = np.array('timeGroup', dtype='S')


def write_fields(f, E=None, B=None, sources=None):
    if E is not None:
        write_scalar(f, E['y'].trim(), 'Ey')
        write_scalar(f, B['z'].trim(), 'Bz')
        write_scalar(f, E['z'].trim(), 'Ez')
    if B is not None:
        write_scalar(f, B['x'].trim(), 'Bx')
        write_scalar(f, E['x'].trim(), 'Ex')
        write_scalar(f, B['y'].trim(), 'By')
    if sources is not None:
        write_scalar(f, sources.rho.trim(), 'rho')
        write_scalar(f, sources.Jx.trim(), 'Jx')
        write_scalar(f, sources.Jy.trim(), 'Jy')
        write_scalar(f, sources.Jz.trim(), 'Jz')


def write_particles(f, ions):
    dset = f.create_dataset('ions', (ions.N, 5), dtype=np.float64)
    dset[:, :] = ions.view('(5,)f8')[:ions.N, :]
    dset.attrs['vsType'] = np.array('variableWithMesh', dtype='S')
    dset.attrs['vsNumSpatialDims'] = 2
    dset.attrs['vsLabels'] = np.array('x, y, vx, vy, vz', dtype='S')
