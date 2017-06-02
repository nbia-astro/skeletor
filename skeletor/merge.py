import numpy as np
import h5py

snap = 0
size = 4


def merge_files(snap, size):
    filenames = ['data/id{}/fields{}.h5'.format(rank, snap)
                 for rank in range(size)]

    f = h5py.File(filenames[0], 'r')
    (ny, nx) = f['grid'].attrs['ny_nx']
    (lby, lbx) = f['grid'].attrs['lby_lbx']
    (uby, ubx) = f['grid'].attrs['uby_ubx']
    size = f['grid'].attrs['size']
    nyp = ny//size
    assert size == len(filenames)
    with_ghost = f['grid'].attrs['with_ghost']

    # Create new file
    f_merged = h5py.File('data/fields{}.h5'.format(snap), 'w')

    # Copy the timegroup
    f.copy('timeGroup', f_merged)

    # Create the grid
    group = f_merged.create_group('grid')
    group.attrs['vsType'] = np.array('mesh', dtype='S')
    group.attrs['vsKind'] = np.array('uniform', dtype='S')
    group.attrs['vsLowerBounds'] = [0.0, 0.0]
    group.attrs['vsUpperBounds'] = f['grid'].attrs['Ly_Lx']
    group.attrs['ny_nx'] = f['grid'].attrs['ny_nx']
    group.attrs['Ly_Lx'] = f['grid'].attrs['Ly_Lx']
    group.attrs['dy_dx'] = f['grid'].attrs['dy_dx']
    group.attrs['vsNumCells'] = [ny-1, nx-1]

    # Copy all the data from the various snapshots
    fields = ['Bx', 'By', 'Bz', 'Ex', 'Ey', 'Ez', 'Jx', 'Jy', 'Jz', 'rho']
    for field in fields:
        dset = f_merged.create_dataset(field, (ny, nx), dtype=np.float64)
        for filename in filenames:
            f = h5py.File(filename, 'r')
            rank = f['grid'].attrs['rank']

            if with_ghost == b'True':
                dset[nyp*rank:nyp*(rank+1), :nx] = f[field][lby:uby, lbx:ubx]
            else:
                dset[nyp*rank:nyp*(rank+1), :nx] = f[field][:, :]
            f.close()
        dset.attrs['vsType'] = np.array('variable', dtype='S')
        dset.attrs['vsMesh'] = np.array('grid', dtype='S')
        dset.attrs['vsTimeGroup'] = np.array('timeGroup', dtype='S')

    f_merged.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import warnings
    from matplotlib.cbook import mplDeprecation
    from reader import read_grid, read_fields
    plt.rc('image', aspect='auto', interpolation='nearest', origin='lower')

    snaps = 7

    for snap in range(snaps):
        merge_files(snap, 2)

    # Read grid
    grid = read_grid('data/fields{}.h5'.format(0))
    # Read first snapshot
    f, h = read_fields('data/fields{}.h5'.format(0))

    # Create figure
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1)
    im = axes.imshow(f['By'], extent=(0, grid.Lx, 0, grid.Ly))

    # Plot as a function of time
    for snap in range(1, snaps):
        f, h = read_fields('data/fields{}.h5'.format(snap))
        im.set_data(f['By'])
        axes.set_title('t = {:1.2f}'.format(h['t']))
        im.autoscale()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                    "ignore", category=mplDeprecation)
            plt.pause(1e-7)
