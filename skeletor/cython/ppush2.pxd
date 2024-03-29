from types cimport real_t, complex_t

cdef extern from "../../picksc/ppic2/ppush2.h":

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
