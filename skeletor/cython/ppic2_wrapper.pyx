# distutils: sources = picksc/ppic2/pplib2.c picksc/ppic2/ppush2.c

from ctypes cimport complex_t, complex2_t, float_t, float2_t, particle_t
from dtypes import Float, Int
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
cdef int ipbc = 1
# Not sure what this is for
cdef int ntpose = 1


cdef class grid_t(object):
    """Grid extension type.
    This is inherited by the Grid class (see grid.py)."""
    cdef public int nx, ny
    cdef public int kstrt, nvp
    cdef public float_t edges[2]
    cdef public int nyp, noff, nypmx, nypmn


def cppinit(MPI.Comm comm):
    """Start parallel processing"""
    from mpi4py.MPI import Is_initialized
    cdef int idproc, nvp
    pplib2.cppinit2(&idproc, &nvp, comm.ob_mpi, 0, NULL)
    # Make sure MPI is indeed initialized
    assert Is_initialized()
    # 'idproc' and 'nvp' are simply the MPI rank and size. Make sure that this
    # is indeed the case
    assert comm.rank == idproc
    assert comm.size == nvp
    return idproc, nvp


def cppexit():
    """End parallel processing"""
    from mpi4py.MPI import Is_finalized
    pplib2.cppexit()
    assert Is_finalized()


def cpdicomp(int ny, int kstrt, int nvp):
    # edges[0] = lower boundary of particle partition
    # edges[1] = upper boundary of particle partition
    cdef float_t edges[2]
    # nyp = number of primary (complete) gridpoints in particle partition
    # noff = lowermost global gridpoint in particle partition
    # nypmx = maximum size of particle partition, including guard cells
    # nypmn = minimum value of nyp
    cdef int nyp, noff, nypmx, nypmn
    ppush2.cpdicomp2l(
            &edges[0], &nyp, &noff, &nypmx, &nypmn, ny, kstrt, nvp, idps)
    return edges, nyp, noff, nypmx, nypmn


def cppgpush2l(
        particle_t[:] particles, float2_t[:,:] fxy,
        int npp, int[:] ihole, float_t qm, float_t dt, grid_t grid):

    cdef float_t ek = 0.0
    cdef int npmax = particles.shape[0]
    cdef int nxe = fxy.shape[1]
    cdef int ntmax = ihole.shape[0] - 1

    ppush2.cppgpush2l(
            &particles[0].x, &fxy[0,0].x, grid.edges, npp, grid.noff,
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

    pplib2.cppmove2(
            &particles[0].x, grid.edges, &npp,
            &sbufr[0].x, &sbufl[0].x, &rbufr[0].x, &rbufl[0].x, &ihole[0],
            grid.ny, grid.kstrt, grid.nvp, idimp, npmax, idps, nbmax, ntmax,
            &info[0])

    return npp

def cppgpost2l(
        particle_t[:] particles, float_t[:,:] rho, int np, int noff,
        float_t charge, int npmax):

    cdef int nxe = rho.shape[1]
    cdef int nypmx = rho.shape[0]

    ppush2.cppgpost2l(&particles[0].x, &rho[0,0], np, noff, charge, idimp,
            npmax, nxe, nypmx)

def cppaguard2xl(float_t[:,:] rho, grid_t grid):

    cdef int nxe = rho.shape[1]

    ppush2.cppaguard2xl(&rho[0,0], grid.nyp, grid.nx, nxe, grid.nypmx)


def cppnaguard2l(
        float_t[:,:] rho, float_t[:,:] scr, grid_t grid):

    cdef int nxe = rho.shape[1]

    pplib2.cppnaguard2l(
            &rho[0,0], &scr[0,0], grid.nyp, grid.nx,
            grid.kstrt, grid.nvp, nxe, grid.nypmx)


def cppcguard2xl(float2_t[:,:] fxy, grid_t grid):

    cdef int nxe = fxy.shape[1]

    ppush2.cppcguard2xl(&fxy[0,0].x, grid.nyp, grid.nx, ndim, nxe, grid.nypmx)


def cppncguard2l(float2_t[:,:] fxy, grid_t grid):

    cdef int nxe = fxy.shape[1]

    pplib2.cppncguard2l(
            &fxy[0,0].x, grid.nyp, grid.kstrt, grid.nvp, 2*nxe, grid.nypmx)

def cwpfft2rinit(int[:] mixup, complex_t[:] sct, int indx, int indy):

    cdef int nxhy = mixup.shape[0]
    cdef int nxyh = sct.shape[0]

    ppush2.cwpfft2rinit(&mixup[0], &sct[0], indx, indy, nxhy, nxyh)

def cwppfft2r(
        float_t[:,:] qe, complex_t[:,:] qt,
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

    cdef float_t ttp = 0.0

    ppush2.cwppfft2r(
        <complex_t *>&qe[0,0], &qt[0,0], &bs[0,0].x, &br[0,0].x,
        isign, ntpose, &mixup[0], &sct[0], &ttp, indx, indy,
        grid.kstrt, grid.nvp, nxe//2, nye, kxp, kyp, nypmx, nxhy, nxyh)

    return ttp

def cppois22(
        complex_t[:,:] qt, complex2_t[:,:] fxyt,
        int isign, complex_t[:,:] ffc, float_t ax, float_t ay, float_t affp,
        grid_t grid):

    cdef int nye = qt.shape[1]
    cdef int kxp = qt.shape[0]
    cdef int nyh = ffc.shape[1]

    cdef float_t we = 0.0

    ppush2.cppois22(
            &qt[0,0], &fxyt[0,0].x, isign, &ffc[0,0], ax, ay, affp,
            &we, grid.nx, grid.ny, grid.kstrt, nye, kxp, nyh)

    return we

def cwppfft2r2(
        float2_t[:,:] fxye, complex2_t[:,:] fxyt,
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

    cdef float_t ttp = 0.0

    ppush2.cwppfft2r2(
            <complex_t *>&fxye[0,0].x, &fxyt[0,0].x, &bs[0,0].x, &br[0,0].x,
            isign, ntpose, &mixup[0], &sct[0], &ttp, indx, indy,
            grid.kstrt, grid.nvp, nxe//2, nye, kxp, kyp, nypmx, nxhy, nxyh)

    return ttp

def cppdsortp2yl(
        particle_t[:] particles, particle_t[:] particles2,
        int[:] npic, int npp, grid_t grid):

    cdef int npmax = particles.shape[0]

    ppush2.cppdsortp2yl(
            &particles[0].x, &particles2[0].x, &npic[0],
            npp, grid.noff, grid.nyp, idimp, npmax, grid.nypmx)

def cpdistr2(particle_t[:] particles, float_t vtx, float_t vty,
        float_t vdx, float_t vdy, int npx, int npy, grid_t grid):

    cdef int nps = 1
    cdef int npp = 0
    cdef int npmax = particles.shape[0]
    cdef int ierr

    ppush2.cpdistr2(&particles[0].x, grid.edges, &npp, nps,
            vtx, vty, vdx, vdy, npx, npy, grid.nx, grid.ny,
            idimp, npmax, idps, ipbc, &ierr)

    return npp, ierr
