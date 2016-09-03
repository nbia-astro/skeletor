from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
# from skeletor import Poisson as Poisson
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

def ppic2(mpifft= False, indy = 9):
  """Test of twostream instability."""

  if mpifft:
    from skeletor import PoissonMpiFFT4py as Poisson
  else:
    from skeletor import Poisson as Poisson

  nx = 2**9
  ny = 2**indy

  # Start parallel processing
  idproc, nvp = cppinit(comm)

  # Create numerical grid. This contains information about the extent of
  # the subdomain assigned to each processor.
  grid = Grid(nx, ny, comm)

  npx = 3072
  npy = 3072

  npar = npx*npy

  tend = 10.
  dt   = 0.1
  nt   = int(tend/dt)

  vtx = 1.0
  vty = 1.0
  vx0 = 0.0
  vy0 = 0.0

  charge = -1
  mass   = 1

  ax = 0.912871
  ay = 0.912871

  # Maximum number of electrons in each partition
  npmax = int(1.25*npar/nvp)

  # Create particle array
  electrons = Particles(npmax, charge, mass)

  electrons.initialize_ppic2(vtx, vty, vx0, vy0, npx, npy, grid)

  # Make sure the numbers of particles in each subdomain add up to the
  # total number of particles
  # assert comm.allreduce(electrons.np, op=MPI.SUM) == np

  # Set the electric field to zero
  E = Field(grid, comm, dtype=Float2)
  E.fill((0.0, 0.0))

  # Initialize sources
  sources = Sources(grid, comm, dtype=Float)

  # Initialize Poisson solver
  poisson = Poisson(grid, ax, ay, npar)

  # Calculate initial density and force

  # Deposit sources
  sources.deposit_ppic2(electrons)
  sources.rho.add_guards_ppic2()

  # Solve Gauss' law
  poisson(sources.rho, E, destroy_input=False)
  # Set boundary condition
  E.copy_guards_ppic2()

  print('Process {} saying hi!'.format(comm.rank))

  t = 0
  ##########################################################################
  # Main loop over time                                                    #
  ##########################################################################
  # Start the timer
  t_push = 0
  t_deposit = 0
  t_add_guard = 0
  t_copy_guard = 0
  t_solve = 0
  tstart = MPI.Wtime()
  for it in range(nt):

      # Push particles on each processor. This call also sends and
      # receives particles to and from other processors/subdomains.
      t_temp  = MPI.Wtime()
      electrons.push(E, dt)
      t_push += MPI.Wtime() - t_temp

      # Update time
      t += dt

      # Deposit sources
      t_temp  = MPI.Wtime()
      sources.deposit_ppic2(electrons)
      t_deposit += MPI.Wtime() - t_temp

      # Boundary calls
      t_temp  = MPI.Wtime()
      sources.rho.add_guards_ppic2()
      t_add_guard += MPI.Wtime() - t_temp

      # Solve Gauss' law
      t_temp  = MPI.Wtime()
      poisson(sources.rho, E, destroy_input=False)
      t_solve += MPI.Wtime() - t_temp

      # Set boundary condition
      t_temp  = MPI.Wtime()
      E.copy_guards_ppic2()
      t_copy_guard += MPI.Wtime() - t_temp

  # Get elapsed time
  elapsed_time = MPI.Wtime() - tstart
  if comm.rank == 0:
    print("Push time {}".format(t_push))
    print("Deposit time {}".format(t_deposit))
    print("Add guard time {}".format(t_add_guard))
    print("Copy guard time {}".format(t_copy_guard))
    print("Solve Gauss time {}".format(t_solve))
    print("Total time {}".format(elapsed_time))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mpifft', '-mpifft', action='store_true')
    parser.add_argument('indy', type=int)
    args = parser.parse_args()

    ppic2(indy=args.indy)
