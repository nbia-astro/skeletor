from mpi4py.libmpi cimport MPI_Comm

cdef extern from "ppic2/pplib2.h":

    void cppinit2 (int *idproc, int *nvp, MPI_Comm comm,
            int argc, char *argv[])

    void cppexit ()

    void cppabort ()

    void cppmove2 (float part[], float edges[], int *npp, float sbufr[],
            float sbufl[], float rbufr[], float rbufl[], int ihole[],
            int ny, int kstrt, int nvp, int idimp, int npmax, int idps,
            int nbmax, int ntmax, int info[])

    void cppnaguard2l(float f[], float scr[], int nyp, int nx, int kstrt,
            int nvp, int nxv, int nypmx)

    void cppncguard2l(float f[], int nyp, int kstrt, int nvp, int nxv,
            int nypmx)
