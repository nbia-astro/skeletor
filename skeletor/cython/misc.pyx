from types cimport real_t, grid_t
from types import Float


def assemple_arrays(real_t[:] x_cell, real_t[:] y_cell, real_t[:] vx_cell,
                    real_t[:] vy_cell, real_t[:] vz_cell, real_t[:] x,
                    real_t[:] y, real_t[:] vx, real_t[:] vy, real_t[:] vz,
                    const int npc, grid_t grid):

    """
    Initialize arrays with particle positions and velocities for a quiet
    Maxwellian. The result has x and y in grid distance units and the
    velocities in 'physical' units.
    Cython is used to bring down the speed of the triple loop.
    """
    cdef int i, j, k, l
    cdef int N = grid.nx*grid.nyp*npc
    l = 0

    for i in range(grid.nx):
        for j in range(grid.nyp):
            for k in range(npc):
                x[l] = x_cell[k] + i
                y[l] = y_cell[k] + j + grid.noff
                vx[l] = vx_cell[k]
                vy[l] = vy_cell[k]
                vz[l] = vz_cell[k]
                l += 1
