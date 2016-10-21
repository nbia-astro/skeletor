/* header file for pplib2.c */

#include "mpi.h"
#include "precision.h"

void cppinit2(int *idproc, int *nvp, MPI_Comm comm, int argc, char *argv[]);

void cppexit();

void cppabort();

void cpwtimera(int icntrl, real_t *time, double *dtime);

void cppsum(real_t f[], real_t g[], int nxp);

void cppdsum(double f[], double g[], int nxp);

void cppimax(int f[], int g[], int nxp);

void cppdmax(double f[], double g[], int nxp);

void cppncguard2l(real_t f[], int nyp, int kstrt, int nvp, int nxv,
                  int nypmx);

void cppnaguard2l(real_t f[], real_t scr[], int nyp, int nx, int kstrt,
                  int nvp, int nxv, int nypmx);

void cppnacguard2l(real_t f[], real_t scr[], int nyp, int nx, int ndim,
                   int kstrt, int nvp, int nxv, int nypmx);

void cpptpose(complex_t f[], complex_t g[], complex_t s[],
              complex_t t[], int nx, int ny, int kxp, int kyp,
              int kstrt, int nvp, int nxv, int nyv, int kxpd, int kypd);

void cppntpose(complex_t f[], complex_t g[], complex_t s[],
               complex_t t[], int nx, int ny, int kxp, int kyp,
               int kstrt, int nvp, int ndim, int nxv, int nyv, int kxpd,
               int kypd);

void cppmove2(real_t part[], real_t edges[], int *npp, real_t sbufr[],
              real_t sbufl[], real_t rbufr[], real_t rbufl[], int ihole[],
              int ny, int kstrt, int nvp, int idimp, int npmax, int idps,
              int nbmax, int ntmax, int info[]);
