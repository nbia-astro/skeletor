from skeletor import Grid, Field, Float
from mpi4py.MPI import COMM_WORLD as comm
import numpy as np

nx = 64
ny = 32
# Create numerical grid
grid = Grid(nx, ny, comm, nlbx=2, nubx=2, nlby=2, nuby=2)

# Coordinate arrays
xx, yy = np.meshgrid(grid.x, grid.y)

# Initialize density field
rho = Field(grid, comm, dtype=Float)
rho.fill(0.0)

# Extending the grid.
print("Attributes that differ for each rank:\n")
print("rank: {}, noff={}, edges[0]={}, edges[1]={}".
      format(comm.rank, grid.noff, grid.edges[0], grid.edges[1]))

if comm.rank == 0:
    print("\nAttributes that do not differ for each rank:\n")
    print("nyp={}, nypmx={}, nypmn={}".
          format(grid.nyp, grid.nypmx, grid.nypmn))

    print("lby={}, uby={}, nlby={}, nuby={}".
          format(grid.lby, grid.uby, grid.nlby, grid.nuby))
