cdef extern from "ppic2/pplib2.h":

    void cppinit2 (int *idproc, int *nvp, int argc, char *argv[])

    void cppexit ()

    void cppabort ()

    void cppmove2 (float part[], float edges[], int *npp, float sbufr[],
            float sbufl[], float rbufr[], float rbufl[], int ihole[],
            int ny, int kstrt, int nvp, int idimp, int npmax, int idps,
            int nbmax, int ntmax, int info[])
