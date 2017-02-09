from mpi4py.libmpi cimport MPI_Comm
from types cimport real_t

cdef extern from "../../picksc/ppic2/pplib2.h":

    void cppinit2 (int *idproc, int *nvp, MPI_Comm comm,
            int argc, char *argv[])

    void cppexit ()

    void cppabort ()

    void cppmove2 (real_t part[], real_t edges[], int *npp, real_t sbufr[],
            real_t sbufl[], real_t rbufr[], real_t rbufl[], int ihole[],
            int ny, int kstrt, int nvp, int idimp, int npmax, int idps,
            int nbmax, int ntmax, int info[])

    void cppnaguard2l(real_t f[], real_t scr[], int nyp, int nx, int kstrt,
            int nvp, int nxv, int nypmx)

    void cppncguard2l(real_t f[], int nyp, int kstrt, int nvp, int nxv,
            int nypmx)
