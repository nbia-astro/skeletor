cdef extern from "ppic2/ppush2.h":

    void cpdicomp2l (float edges[], int *nyp, int *noff, int *nypmx,
            int *nypmn, int ny, int kstrt, int nvp, int idps)

    void cpdistr2 (float part[], float edges[], int *npp, int nps, float vtx,
            float vty, float vdx, float vdy, int npx, int npy, int nx, int ny,
            int idimp, int npmax, int idps, int ipbc, int *ierr)

    void cppgpush2l (float part[], float fxy[], float edges[], int npp,
            int noff, int ihole[], float qbm, float dt, float *ek,
            int nx, int ny, int idimp, int npmax, int nxv,
            int nypmx, int idps, int ntmax, int ipbc)
