import ppic2_wrapper as parallel
import numpy
from mpi4py import MPI

# MPI communicator
comm = MPI.COMM_WORLD

# Exponents that determines grid points in x- and y-direction
# indx, indy = 9, 9
indx, indy = 5, 5

# Number of electrons distributed in x- and y-direction
# npx, npy = 3072, 3072
npx, npy = 512, 512

# Thermal velocity of electrons in x- and y-direction
vtx, vty = 1.0, 1.0
# Drift velocity of electrons in x- and y-direction
vdx, vdy = 0.0, 0.0

# Number of partition boundaries
idps = 2
# number of particle coordinates
idimp = 4
# ipbc = particle boundary condition: 1 = periodic
ipbc = 1

idproc, nvp = parallel.cppinit ()

# Synchronize random number generator across processes
numpy.random.set_state (comm.bcast (numpy.random.get_state ()))

# Number of grid points in x- and y-direction
nx = 1 << indx
ny = 1 << indy

# Total number of particles in simulation
np = npx*npy

# maximum number of electrons in each partition
npmax = int (1.25*np/nvp)
# Size of buffer for passing particles between processors
nbmax = 0.1*npmax
# Size of ihole buffer for particles leaving processor
ntmax = 2*nbmax

# Don't know why this is useful
kstrt = idproc + 1

if nvp > ny:
    if kstrt == 1:
        msg = "Too many processors requested: ny={}, nvp={}"
        print (msg.format (ny, nvp))
    parallel.cppexit ()

edges, nyp, noff, nypmx, nypmn = parallel.cpdicomp (ny, kstrt, nvp, idps)

# Starting address of particles in partition
nps = 1
# Initialize particle array
part, npp, ierr = parallel.cpdistr (edges, nps, vtx, vty, vdx, vdy,
        npx, npy, nx, ny, idimp, npmax, idps, ipbc)

# Phase space coordinates before drift
x1 = part[0::idimp].copy ()
y1 = part[1::idimp].copy ()
vx1 = part[2::idimp].copy ()
vy1 = part[3::idimp].copy ()

# Make sure particles actually reside in the local subdomain
assert all (y1[:npp] >= edges[0]) and all (y1[:npp] < edges[1])

# Library drift
dt = 0.1
nxe = nx + 2
ihole, ek = parallel.cppgpush (part, edges, npp, noff, dt, nx, ny, idimp,
        npmax, nxe, nypmx, idps, ntmax, ipbc)
# Check for ihole overflow error
if ihole[0] < 0:
    ierr = -ihole[0]
    print ("ihole overflow error: ntmax={}, ih={}".format (ntmax, ierr))

# Phase space coordinates after drift
x2 = part[0::idimp].copy ()
y2 = part[1::idimp].copy ()
vx2 = part[2::idimp].copy ()
vy2 = part[3::idimp].copy ()

# Manual drift
x3 = numpy.empty_like (x1)
y3 = numpy.empty_like (y1)
x3[:npp] = x1[:npp] + vx1[:npp]*dt
y3[:npp] = y1[:npp] + vy1[:npp]*dt
# Apply boundary condition in x
Lx = 1.0*nx
ind = numpy.where (x3[:npp] >= Lx); x3[ind] -= Lx
ind = numpy.where (x3[:npp] < 0.0); x3[ind] += Lx

# Make sure there's agrrement between manual and library drift
assert all (x3[:npp] == x2[:npp])
assert all (y3[:npp] == y2[:npp])

npp_, info = parallel.cppmove (part, edges, npp, ihole, ny, kstrt, nvp, idimp,
        npmax, idps, nbmax, ntmax)
assert comm.reduce (npp, op=MPI.SUM) == comm.reduce (npp_, op=MPI.SUM)

# Phase space coordinates after communication
x4 = part[0::idimp].copy ()
y4 = part[1::idimp].copy ()
vx4 = part[2::idimp].copy ()
vy4 = part[3::idimp].copy ()

# Make sure particles actually reside in the local subdomain
assert all (y4[:npp_] >= edges[0]) and all (y4[:npp_] < edges[1])

parallel.cppexit ()
