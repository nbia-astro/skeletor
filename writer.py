import numpy as np

# Example of writing hdf5 fs in the VizSchema format
# See https://ice.txcorp.com/trac/vizschema

# Parallel HDF5 can be installed by doing the following
# brew install hdf5 --with-mpi
# brew install h5utils
# The next two lines as one bash command
# CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/usr/local/Cellar/hdf5/1.10.0-patch1
# pip install --no-binary=h5py h5py


def write_grid(f, grid, write_ghosts=False):
    # These shold be defined already..
    ymin = grid.noff*grid.dy
    ymax = (grid.noff + grid.nyp)*grid.dy
    xmin = 0.
    xmax = grid.Lx

    group = f.create_group('grid')
    group.attrs['vsType'] = np.array('mesh', dtype='S')
    group.attrs['vsKind'] = np.array('uniform', dtype='S')
    if write_ghosts:
        group.attrs['vsNumCells'] = [grid.myp-1, grid.mx-1]
        ymin -= grid.lby*grid.dy
        ymax += grid.lby*grid.dy
        xmin -= grid.lbx*grid.dx
        xmax += grid.lbx*grid.dx
        group.attrs['with_ghost'] = np.array('True', dtype='S')
    else:
        group.attrs['vsNumCells'] = [grid.nyp-1, grid.nx-1]
        group.attrs['with_ghost'] = np.array('False', dtype='S')
    group.attrs['vsLowerBounds'] = [ymin, xmin]
    group.attrs['vsUpperBounds'] = [ymax, xmax]

    group.attrs['ny_nx'] = [grid.ny, grid.nx]
    group.attrs['lby_lbx'] = [grid.lby, grid.lbx]
    group.attrs['uby_ubx'] = [grid.uby, grid.ubx]
    group.attrs['Ly_Lx'] = [grid.Ly, grid.Lx]
    group.attrs['dy_dx'] = [grid.dy, grid.dx]
    group.attrs['rank'] = grid.comm.rank
    group.attrs['size'] = grid.comm.size


def write_time(f, t, it):
    group = f.create_group('timeGroup')
    group.attrs['vsKind'] = np.array('time', dtype='S')
    group.attrs['vsTime'] = t
    group.attrs['vsStep'] = it


def write_scalar(f, scalar, name, write_ghosts=False):
    if write_ghosts:
        phi = f.create_dataset(name, data=scalar)
    else:
        phi = f.create_dataset(name, data=scalar.trim())
    phi.attrs['vsType'] = np.array('variable', dtype='S')
    phi.attrs['vsMesh'] = np.array('grid', dtype='S')
    phi.attrs['vsTimeGroup'] = np.array('timeGroup', dtype='S')


def write_fields(f, E=None, B=None, sources=None, write_ghosts=False):
    if E is not None:
        write_scalar(f, E['y'], 'Ey', write_ghosts)
        write_scalar(f, B['z'], 'Bz', write_ghosts)
        write_scalar(f, E['z'], 'Ez', write_ghosts)
    if B is not None:
        write_scalar(f, B['x'], 'Bx', write_ghosts)
        write_scalar(f, E['x'], 'Ex', write_ghosts)
        write_scalar(f, B['y'], 'By', write_ghosts)
    if sources is not None:
        write_scalar(f, sources.rho, 'rho', write_ghosts)
        write_scalar(f, sources.Jx, 'Jx', write_ghosts)
        write_scalar(f, sources.Jy, 'Jy', write_ghosts)
        write_scalar(f, sources.Jz, 'Jz', write_ghosts)


def write_particles(f, ions):
    dset = f.create_dataset('ions', (ions.N, 5), dtype=np.float64)
    dset[:, :] = ions.view('(5,)f8')[:ions.N, :]
    dset.attrs['vsType'] = np.array('variableWithMesh', dtype='S')
    dset.attrs['vsNumSpatialDims'] = 2
    dset.attrs['vsLabels'] = np.array('x, y, vx, vy, vz', dtype='S')
