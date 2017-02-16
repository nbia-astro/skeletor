# distutils: sources = picksc/ppic2/pplib2.c picksc/ppic2/ppush2.c

from types cimport complex_t, complex2_t, real_t, real2_t, particle_t, grid_t
from types import Float, Int
cimport pplib2, ppush2
from numpy cimport ndarray
cimport mpi4py.MPI as MPI
import numpy


# See https://bitbucket.org/mpi4py/mpi4py/issues/1/mpi4py-cython-openmpi
cdef extern from 'mpi-compat.h': pass


# Number of velocity coordinates
cdef int ndim = 2
# Number of partition boundaries
cdef int idps = 2
# Number of particle phase space coordinates
cdef int idimp = 4
# Particle boundary condition: 1 = periodic
cdef int ipbc = 0
# Not sure what this is for
cdef int ntpose = 1

# Processor ID and number of processors (to be set in cppinit below)
cdef int idproc, nvp


def cppinit(MPI.Comm comm):
    """Start parallel processing"""
    from mpi4py.MPI import Is_initialized
    global idproc, nvp
    pplib2.cppinit2(&idproc, &nvp, comm.ob_mpi, 0, NULL)
    # Make sure MPI is indeed initialized
    assert Is_initialized()
    # 'idproc' and 'nvp' are simply the MPI rank and size. Make sure that this
    # is indeed the case
    assert idproc == comm.Get_rank()
    assert nvp == comm.Get_size()
    return idproc, nvp


def cppexit():
    """End parallel processing"""
    from mpi4py.MPI import Is_finalized
    pplib2.cppexit()
    assert Is_finalized()


def cppgbpush2l(
        particle_t[:] particles, real2_t[:,:] fxy, real_t bz,
        int npp, int[:] ihole, real_t qm, real_t dt, grid_t grid):

    cdef real_t ek = 0.0
    cdef int npmax = particles.shape[0]
    cdef int nxe = fxy.shape[1]
    cdef int ntmax = ihole.shape[0] - 1

    ppush2.cppgbpush2l(
            &particles[0].x, &fxy[0,0].x, bz, grid.edges, npp, grid.noff,
            &ihole[0], qm, dt, &ek, grid.nx, grid.ny, idimp, npmax, nxe,
            grid.nypmx, idps, ntmax, ipbc)

    return ek


def cppmove2(
        particle_t[:] particles, int npp,
        particle_t[:] sbufl, particle_t[:] sbufr,
        particle_t[:] rbufl, particle_t[:] rbufr,
        int[:] ihole, int[:] info, grid_t grid):

    cdef int npmax = particles.shape[0]
    cdef int nbmax = sbufl.shape[0]
    cdef int ntmax = ihole.shape[0] - 1
    cdef int kstrt = idproc + 1

    pplib2.cppmove2(
            &particles[0].x, grid.edges, &npp,
            &sbufr[0].x, &sbufl[0].x, &rbufr[0].x, &rbufl[0].x, &ihole[0],
            grid.ny, kstrt, nvp, idimp, npmax, idps, nbmax, ntmax, &info[0])

    return npp

def cppgpost2l(
        particle_t[:] particles, real_t[:,:] rho, int np, int noff,
        real_t charge, int npmax):

    cdef int nxe = rho.shape[1]
    cdef int nypmx = rho.shape[0]

    ppush2.cppgpost2l(&particles[0].x, &rho[0,0], np, noff, charge, idimp,
            npmax, nxe, nypmx)

def cwpfft2rinit(int[:] mixup, complex_t[:] sct, int indx, int indy):

    cdef int nxhy = mixup.shape[0]
    cdef int nxyh = sct.shape[0]

    ppush2.cwpfft2rinit(&mixup[0], &sct[0], indx, indy, nxhy, nxyh)

def cwppfft2r(
        real_t[:,:] qe, complex_t[:,:] qt,
        complex2_t[:,:] bs, complex2_t[:,:] br,
        int isign, int[:] mixup, complex_t[:] sct,
        int indx, int indy, grid_t grid):

    cdef int nxe = qe.shape[1]
    cdef int nypmx = qe.shape[0]
    cdef int nye = qt.shape[1]
    cdef int kxp = bs.shape[1]
    cdef int kyp = bs.shape[0]
    cdef int nxhy = mixup.shape[0]
    cdef int nxyh = sct.shape[0]
    cdef int kstrt = idproc + 1

    cdef real_t ttp = 0.0

    ppush2.cwppfft2r(
        <complex_t *>&qe[0,0], &qt[0,0], &bs[0,0].x, &br[0,0].x,
        isign, ntpose, &mixup[0], &sct[0], &ttp, indx, indy,
        kstrt, nvp, nxe//2, nye, kxp, kyp, nypmx, nxhy, nxyh)

    return ttp

def cppois22(
        complex_t[:,:] qt, complex2_t[:,:] fxyt,
        int isign, complex_t[:,:] ffc, real_t ax, real_t ay, real_t affp,
        grid_t grid):

    cdef int nye = qt.shape[1]
    cdef int kxp = qt.shape[0]
    cdef int nyh = ffc.shape[1]
    cdef int kstrt = idproc + 1

    cdef real_t we = 0.0

    ppush2.cppois22(
            &qt[0,0], &fxyt[0,0].x, isign, &ffc[0,0], ax, ay, affp,
            &we, grid.nx, grid.ny, kstrt, nye, kxp, nyh)

    return we

def cwppfft2r2(
        real2_t[:,:] fxye, complex2_t[:,:] fxyt,
        complex2_t[:,:] bs, complex2_t[:,:] br,
        int isign, int[:] mixup, complex_t[:] sct,
        int indx, int indy, grid_t grid):

    cdef int nxe = fxye.shape[1]
    cdef int nypmx = fxye.shape[0]
    cdef int nye = fxyt.shape[1]
    cdef int kxp = bs.shape[1]
    cdef int kyp = bs.shape[0]
    cdef int nxhy = mixup.shape[0]
    cdef int nxyh = sct.shape[0]
    cdef int kstrt = idproc + 1

    cdef real_t ttp = 0.0

    ppush2.cwppfft2r2(
            <complex_t *>&fxye[0,0].x, &fxyt[0,0].x, &bs[0,0].x, &br[0,0].x,
            isign, ntpose, &mixup[0], &sct[0], &ttp, indx, indy,
            kstrt, nvp, nxe//2, nye, kxp, kyp, nypmx, nxhy, nxyh)

    return ttp

def cppdsortp2yl(
        particle_t[:] particles, particle_t[:] particles2,
        int[:] npic, int npp, grid_t grid):

    cdef int npmax = particles.shape[0]

    ppush2.cppdsortp2yl(
            &particles[0].x, &particles2[0].x, &npic[0],
            npp, grid.noff, grid.nyp, idimp, npmax, grid.nypmx)

def cpdistr2(particle_t[:] particles, real_t vtx, real_t vty,
        real_t vdx, real_t vdy, int npx, int npy, grid_t grid):

    cdef int nps = 1
    cdef int npp = 0
    cdef int npmax = particles.shape[0]
    cdef int ierr

    ppush2.cpdistr2(&particles[0].x, grid.edges, &npp, nps,
            vtx, vty, vdx, vdy, npx, npy, grid.nx, grid.ny,
            idimp, npmax, idps, ipbc, &ierr)

    return npp, ierr
