from types cimport real_t, complex_t

cdef extern from "../../picksc/ppic2/ppush2.h":

    void cpdicomp2l (real_t edges[], int *nyp, int *noff, int *nypmx,
            int *nypmn, int ny, int kstrt, int nvp, int idps)

    void cpdistr2 (real_t part[], real_t edges[], int *npp, int nps, real_t vtx,
            real_t vty, real_t vdx, real_t vdy, int npx, int npy, int nx, int ny,
            int idimp, int npmax, int idps, int ipbc, int *ierr)

    void cppgpush2l (real_t part[], real_t fxy[], real_t edges[], int npp,
            int noff, int ihole[], real_t qbm, real_t dt, real_t *ek,
            int nx, int ny, int idimp, int npmax, int nxv,
            int nypmx, int idps, int ntmax, int ipbc)

    void cppgbpush2l (real_t part[], real_t fxy[], real_t bz, real_t edges[], int npp,
            int noff, int ihole[], real_t qbm, real_t dt, real_t *ek,
            int nx, int ny, int idimp, int npmax, int nxv,
            int nypmx, int idps, int ntmax, int ipbc)

    void cppgpost2l(real_t part[], real_t q[], int npp, int noff, real_t qm,
            int idimp, int npmax, int nxv, int nypmx)

    void cppaguard2xl(real_t q[], int nyp, int nx, int nxe, int nypmx)

    void cppcguard2xl(real_t fxy[], int nyp, int nx, int ndim, int nxe,
            int nypmx)

    void cwpfft2rinit(int mixup[], complex_t sct[], int indx, int indy,
                      int nxhyd, int nxyhd)

    void cwppfft2r(complex_t f[], complex_t g[], complex_t bs[],
            complex_t br[], int isign, int ntpose, int mixup[],
            complex_t sct[], real_t *ttp, int indx, int indy, int kstrt,
            int nvp, int nxvh, int nyv, int kxp, int kyp, int kypd, int nxhyd,
            int nxyhd)

    void cppois22(complex_t q[], complex_t fxy[], int isign,
            complex_t ffc[], real_t ax, real_t ay, real_t affp, real_t *we,
            int nx, int ny, int kstrt, int nyv, int kxp, int nyhd)

    void cwppfft2r2(complex_t f[], complex_t g[], complex_t bs[],
            complex_t br[], int isign, int ntpose, int mixup[],
            complex_t sct[], real_t *ttp, int indx, int indy, int kstrt,
            int nvp, int nxvh, int nyv, int kxp, int kyp, int kypd, int nxhyd,
            int nxyhd)

    void cppdsortp2yl(real_t parta[], real_t partb[], int npic[], int npp,
            int noff, int nyp, int idimp, int npmax, int nypm1)

    void cpdistr2(real_t part[], real_t edges[], int *npp, int nps, real_t vtx,
            real_t vty, real_t vdx, real_t vdy, int npx, int npy, int nx, int ny,
            int idimp, int npmax, int idps, int ipbc, int *ierr)
