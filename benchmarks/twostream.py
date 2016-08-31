from skeletor import cppinit, Float, Float2, Grid, Field, Particles, Sources
# from skeletor import Poisson as Poisson
import numpy
from mpi4py import MPI
from mpi4py.MPI import COMM_WORLD as comm

def test_twostream(mpifft= False, ny = 8):
  """Test of twostream instability."""

  if mpifft:
    from skeletor import PoissonMpiFFT4py as Poisson
  else:
    from skeletor import Poisson as Poisson

  # Get rid of this one
  A = 0.01

  quiet = True

  # Number of grid points in x-direction
  nx = 64

  # Size of box
  Lx, Ly = nx, ny

  # Average number of particles per cell
  npc = 32

  # Number of time steps
  nt = 20

  # Background ion density
  n0 = 1.0

  assert(numpy.sqrt(npc/2) % 1 == 0)
  # Particle charge and mass
  charge = -1.0
  mass = 1.0

  # Timestep
  dt = 0.02

  # Smoothed particle size in x/y direction
  ax = 0
  ay = 0

  # Total number of particles in simulation
  npar = npc*nx*ny

  nmode = 1

  kx = 2*numpy.pi/Lx

  # Mean velocity of electrons in x- and y-direction
  vdx = 6
  vdy = 0

  # Thermal velocity of electrons in x- and y-direction
  vtx, vty = 1e-8, 1e-8


  if quiet:
      # Uniform distribution of particle positions (quiet start)
      sqrt_npc = int(numpy.sqrt(npc/2))
      assert sqrt_npc**2 == npc/2
      dx = dy = 1/sqrt_npc
      x, y = numpy.meshgrid(
              numpy.arange(0, nx, dx),
              numpy.arange(0, ny, dy))
      x = x.flatten()
      y = y.flatten()
  else:
      x = nx*numpy.random.uniform(size=np).astype(Float)
      y = ny*numpy.random.uniform(size=np).astype(Float)

  # Normal distribution of particle velocities
  vx = vdx*numpy.ones_like(x)
  vy = vdy*numpy.ones_like(y)

  # Have two particles at position
  x = numpy.concatenate([x,x])
  y = numpy.concatenate([y,y])

  # Make counterpropagating in x
  vx = numpy.concatenate([vx,-vx])
  vy = numpy.concatenate([vy,vy])

  # Add thermal component
  vx += vtx*numpy.random.normal(size=npar).astype(Float)
  vy += vty*numpy.random.normal(size=npar).astype(Float)

  # Start parallel processing
  idproc, nvp = cppinit(comm)

  # Create numerical grid. This contains information about the extent of
  # the subdomain assigned to each processor.
  grid = Grid(nx, ny, comm)

  # Maximum number of electrons in each partition
  npmax = int(1.5*npar/nvp)

  # Create particle array
  electrons = Particles(npmax, charge, mass)

  # Assign particles to subdomains
  electrons.initialize(x, y, vx, vy, grid)

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
  sources.rho += n0*npc

  # Solve Gauss' law
  poisson(sources.rho, E, destroy_input=False)
  # Set boundary condition
  E.copy_guards_ppic2()

  t = 0
  ##########################################################################
  # Main loop over time                                                    #
  ##########################################################################
  # Start the timer
  tstart = MPI.Wtime()
  for it in range(nt):

      # Push particles on each processor. This call also sends and
      # receives particles to and from other processors/subdomains.
      electrons.push(E, dt)

      # Update time
      t += dt

      # Deposit sources
      sources.deposit_ppic2(electrons)

      # Boundary calls
      sources.rho.add_guards_ppic2()
      sources.rho += n0

      # Solve Gauss' law
      poisson(sources.rho, E, destroy_input=False)

      # Set boundary condition
      E.copy_guards_ppic2()

  # Get elapsed time
  elapsed_time = MPI.Wtime() - tstart
  if comm.rank == 0:
    f = open('scaling.txt','a')
    f.write('{}\t{}\n'.format(comm.Get_size(), elapsed_time))
    f.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mpifft', '-mpifft', action='store_true')
    parser.add_argument('ny', type=int)
    args = parser.parse_args()

    test_twostream(ny=args.ny)