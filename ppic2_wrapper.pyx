# distutils: sources = ppic2/pplib2.c ppic2/ppush2.c

cimport pplib2, ppush2
from numpy cimport ndarray
import numpy

ctypedef float float_t
ctypedef float complex complex_t

Int = numpy.dtype ("i{}".format (sizeof (int)))
Float = numpy.dtype ("f{}".format (sizeof (float_t)))
Complex = numpy.dtype ("c{}".format (sizeof (complex_t)))

def cppinit ():
    cdef int argc = 1
    cdef char *argv = "ppic2"
    cdef int idproc, nvp
    pplib2.cppinit2 (&idproc, &nvp, argc, &argv)
    return idproc, nvp

def cppexit ():
    pplib2.cppexit ()

def cpdicomp (int ny, int kstrt, int nvp, int idps):
    # edges[0] = lower boundary of particle partition
    # edges[1] = upper boundary of particle partition
    cdef ndarray[float_t,ndim=1] edges = numpy.empty (idps, Float)
    # nyp = number of primary (complete) gridpoints in particle partition
    # noff = lowermost global gridpoint in particle partition
    # nypmx = maximum size of particle partition, including guard cells
    # nypmn = minimum value of nyp
    cdef int nyp, noff, nypmx, nypmn
    ppush2.cpdicomp2l (&edges[0], &nyp, &noff, &nypmx, &nypmn,
            ny, kstrt, nvp, idps)
    return edges, nyp, noff, nypmx, nypmn

def cpdistr (float_t[:] edges, int nps, float_t vtx, float_t vty,
        float_t vdx, float_t vdy, int npx, int npy, int nx, int ny,
        int idimp, int npmax, int idps, int ipbc):
    # Particle array
    cdef ndarray[float_t,ndim=1] part = numpy.empty (idimp*npmax, Float)
    cdef int npp, ierr
    npp = 0
    ppush2.cpdistr2 (&part[0], &edges[0], &npp, nps, vtx, vty, vdx, vdy,
            npx, npy, nx, ny, idimp, npmax, idps, ipbc, &ierr)
    return part, npp, ierr

def cppgpush (float_t[:] part, float_t[:] edges, int npp, int noff,
        float dt, int nx, int ny, int idimp, int npmax,
        int nxv, int nypmx, int idps, int ntmax, int ipbc):
    cdef int ndim = 2
    cdef ndarray[float_t,ndim=1] fxy = numpy.zeros (ndim*nxv*nypmx, Float)
    cdef ndarray[int,ndim=1] ihole = numpy.empty (ntmax+1, Int)
    cdef float_t qbm = 0.0
    cdef float_t ek = 0.0
    ppush2.cppgpush2l (&part[0], &fxy[0], &edges[0], npp,
            noff, &ihole[0], qbm, dt, &ek, nx, ny, idimp, npmax, nxv, nypmx,
            idps, ntmax, ipbc)
    return ihole, ek

def cppmove (float_t[:] part, float_t[:] edges, int npp, int[:] ihole,
        int ny, int kstrt, int nvp, int idimp, int npmax, int idps,
        int nbmax, int ntmax):
    cdef ndarray[float_t,ndim=1] sbufl = numpy.zeros (idimp*nbmax, Float)
    cdef ndarray[float_t,ndim=1] sbufr = numpy.zeros (idimp*nbmax, Float)
    cdef ndarray[float_t,ndim=1] rbufl = numpy.zeros (idimp*nbmax, Float)
    cdef ndarray[float_t,ndim=1] rbufr = numpy.zeros (idimp*nbmax, Float)
    cdef ndarray[int,ndim=1] info = numpy.zeros (7, Int)
    pplib2.cppmove2 (&part[0], &edges[0], &npp,
            &sbufr[0], &sbufl[0], &rbufr[0], &rbufl[0], &ihole[0],
            ny, kstrt, nvp, idimp, npmax, idps, nbmax, ntmax, &info[0])
    return npp, info
