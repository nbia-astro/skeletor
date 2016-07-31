cdef extern from "../../picksc/ppic2/ppush2.h":

    void cpdicomp2l (float edges[], int *nyp, int *noff, int *nypmx,
            int *nypmn, int ny, int kstrt, int nvp, int idps)

    void cpdistr2 (float part[], float edges[], int *npp, int nps, float vtx,
            float vty, float vdx, float vdy, int npx, int npy, int nx, int ny,
            int idimp, int npmax, int idps, int ipbc, int *ierr)

    void cppgpush2l (float part[], float fxy[], float edges[], int npp,
            int noff, int ihole[], float qbm, float dt, float *ek,
            int nx, int ny, int idimp, int npmax, int nxv,
            int nypmx, int idps, int ntmax, int ipbc)

    void cppgpost2l(float part[], float q[], int npp, int noff, float qm,
            int idimp, int npmax, int nxv, int nypmx)

    void cppaguard2xl(float q[], int nyp, int nx, int nxe, int nypmx)

    void cppcguard2xl(float fxy[], int nyp, int nx, int ndim, int nxe,
            int nypmx)

    void cwpfft2rinit(int mixup[], float complex sct[], int indx, int indy,
                      int nxhyd, int nxyhd)

    void cwppfft2r(float complex f[], float complex g[], float complex bs[],
            float complex br[], int isign, int ntpose, int mixup[],
            float complex sct[], float *ttp, int indx, int indy, int kstrt,
            int nvp, int nxvh, int nyv, int kxp, int kyp, int kypd, int nxhyd,
            int nxyhd)

    void cppois22(float complex q[], float complex fxy[], int isign,
            float complex ffc[], float ax, float ay, float affp, float *we,
            int nx, int ny, int kstrt, int nyv, int kxp, int nyhd)

    void cwppfft2r2(float complex f[], float complex g[], float complex bs[],
            float complex br[], int isign, int ntpose, int mixup[],
            float complex sct[], float *ttp, int indx, int indy, int kstrt,
            int nvp, int nxvh, int nyv, int kxp, int kyp, int kypd, int nxhyd,
            int nxyhd)

    void cppdsortp2yl(float parta[], float partb[], int npic[], int npp,
            int noff, int nyp, int idimp, int npmax, int nypm1)

    void cpdistr2(float part[], float edges[], int *npp, int nps, float vtx,
            float vty, float vdx, float vdy, int npx, int npy, int nx, int ny,
            int idimp, int npmax, int idps, int ipbc, int *ierr)