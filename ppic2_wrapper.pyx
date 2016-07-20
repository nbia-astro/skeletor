# distutils: sources = ppic2/pplib2.c ppic2/ppush2.c

from ctypes cimport float_t, force_t, particle_t
from dtypes import Float, Int
cimport pplib2, ppush2
from numpy cimport ndarray
from mpi4py import MPI
import numpy


def cppinit(comm=MPI.COMM_WORLD):
    """Start parallel processing"""
    from mpi4py.MPI import Is_initialized
    cdef int idproc, nvp
    pplib2.cppinit2(&idproc, &nvp, 0, NULL)
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


def cpdicomp(int ny, int kstrt, int nvp, int idps):
    # edges[0] = lower boundary of particle partition
    # edges[1] = upper boundary of particle partition
    cdef ndarray[float_t, ndim=1] edges = numpy.empty(idps, Float)
    # nyp = number of primary (complete) gridpoints in particle partition
    # noff = lowermost global gridpoint in particle partition
    # nypmx = maximum size of particle partition, including guard cells
    # nypmn = minimum value of nyp
    cdef int nyp, noff, nypmx, nypmn
    ppush2.cpdicomp2l(
            &edges[0], &nyp, &noff, &nypmx, &nypmn, ny, kstrt, nvp, idps)
    return edges, nyp, noff, nypmx, nypmn


def cppgpush2l(
        particle_t[:] particles, float_t[:] edges, int npp, int noff,
        float dt, int nx, int ny, int idimp, int npmax, int nxv,
        int nypmx, int idps, int ntmax, int ipbc):
    cdef int ndim = 2
    # Set the force to zero (this will of course change in the future).
    # TODO: Pass this array as argument.
    cdef ndarray[float_t, ndim=1] fxy = numpy.zeros(ndim*nxv*nypmx, Float)
    cdef ndarray[int, ndim=1] ihole = numpy.empty(ntmax+1, Int)
    cdef float_t qbm = 0.0
    cdef float_t ek = 0.0
    ppush2.cppgpush2l(
            &particles[0].x, &fxy[0], &edges[0], npp, noff, &ihole[0], qbm, dt,
            &ek, nx, ny, idimp, npmax, nxv, nypmx, idps, ntmax, ipbc)
    return ihole, ek


def cppmove2(
        particle_t[:] particles, float_t[:] edges, int npp,
        int[:] ihole, int ny, int kstrt, int nvp, int idimp,
        int npmax, int idps, int nbmax, int ntmax):
    # TODO: Allocate these arrays once and for all in the Particle class
    # and pass them as arguments.
    cdef ndarray[float_t, ndim=1] sbufl = numpy.zeros(idimp*nbmax, Float)
    cdef ndarray[float_t, ndim=1] sbufr = numpy.zeros(idimp*nbmax, Float)
    cdef ndarray[float_t, ndim=1] rbufl = numpy.zeros(idimp*nbmax, Float)
    cdef ndarray[float_t, ndim=1] rbufr = numpy.zeros(idimp*nbmax, Float)
    cdef ndarray[int, ndim=1] info = numpy.zeros(7, Int)
    pplib2.cppmove2(
            &particles[0].x, &edges[0], &npp, &sbufr[0], &sbufl[0], &rbufr[0],
            &rbufl[0], &ihole[0], ny, kstrt, nvp, idimp, npmax, idps, nbmax,
            ntmax, &info[0])
    return npp, info

def cppgpost2l(
        particle_t[:] particles, float_t[:,:] rho, int np, int noff,
        int npmax):

    cdef float_t qme = 1.0;
    cdef int idimp = 4
    cdef int nxe = rho.shape[1]
    cdef int nypmx = rho.shape[0]

    ppush2.cppgpost2l(&particles[0].x, &rho[0,0], np, noff, qme, idimp,
            npmax, nxe, nypmx)

def cppaguard2xl(float_t[:,:] rho, int nyp, int nx):

    cdef int nxe = rho.shape[1]
    cdef int nypmx = rho.shape[0]

    ppush2.cppaguard2xl(&rho[0,0], nyp, nx, nxe, nypmx)


def cppnaguard2l(float_t[:,:] rho, int nyp, int nx, comm=MPI.COMM_WORLD):

    cdef int kstrt = comm.rank + 1
    cdef int nvp = comm.size
    cdef int nxe = rho.shape[1]
    cdef int nypmx = rho.shape[0]

    cdef ndarray[float_t, ndim=2] scr = numpy.zeros((2, nxe), Float)

    pplib2.cppnaguard2l(&rho[0,0], &scr[0,0], nyp, nx, kstrt, nvp, nxe, nypmx)


def cppcguard2xl(force_t[:,:] fxy, int nyp, int nx):

    cdef int nxe = fxy.shape[1]
    cdef int nypmx = fxy.shape[0]

    ppush2.cppcguard2xl(&fxy[0,0].x, nyp, nx, 2, nxe, nypmx)


def cppncguard2l(force_t[:,:] fxy, int nyp, int nx, comm=MPI.COMM_WORLD):

    cdef int kstrt = comm.rank + 1
    cdef int nvp = comm.size
    cdef int nxe = fxy.shape[1]
    cdef int nypmx = fxy.shape[0]

    pplib2.cppncguard2l(&fxy[0,0].x, nyp, kstrt, nvp, 2*nxe, nypmx)
