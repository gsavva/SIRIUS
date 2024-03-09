/** \file blas_lapack.h
 *  \brief Interface to some BLAS/LAPACK functions.
 */
#ifndef __BLAS_H__
#define __BLAS_H__

#include <complex>

#define FORTRAN(x) x##_

using ftn_int            = int32_t;
using ftn_len            = int32_t;
using ftn_single         = float;
using ftn_double         = double;
using ftn_complex        = std::complex<float>;
using ftn_double_complex = std::complex<double>;
using ftn_char           = char const *;
using ftn_bool           = bool;

extern "C" {

void FORTRAN(sgemm)(ftn_char            TRANSA,
                    ftn_char            TRANSB,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_int*            K,
                    ftn_single*         ALPHA,
                    ftn_single*         A,
                    ftn_int*            LDA,
                    ftn_single*         B,
                    ftn_int*            LDB,
                    ftn_single*         BETA,
                    ftn_single*         C,
                    ftn_int*            LDC,
                    ftn_len             TRANSA_len,
                    ftn_len             TRANSB_len);

void FORTRAN(dgemm)(ftn_char            TRANSA,
                    ftn_char            TRANSB,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_int*            K,
                    ftn_double*         ALPHA,
                    ftn_double*         A,
                    ftn_int*            LDA,
                    ftn_double*         B,
                    ftn_int*            LDB,
                    ftn_double*         BETA,
                    ftn_double*         C,
                    ftn_int*            LDC,
                    ftn_len             TRANSA_len,
                    ftn_len             TRANSB_len);

void FORTRAN(cgemm)(ftn_char            TRANSA,
                    ftn_char            TRANSB,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_int*            K,
                    ftn_complex*        ALPHA,
                    ftn_complex*        A,
                    ftn_int*            LDA,
                    ftn_complex*        B,
                    ftn_int*            LDB,
                    ftn_complex*        BETA,
                    ftn_complex*        C,
                    ftn_int*            LDC,
                    ftn_len             TRANSA_len,
                    ftn_len             TRANSB_len);

void FORTRAN(zgemm)(ftn_char            TRANSA,
                    ftn_char            TRANSB,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_int*            K,
                    ftn_double_complex* ALPHA,
                    ftn_double_complex* A,
                    ftn_int*            LDA,
                    ftn_double_complex* B,
                    ftn_int*            LDB,
                    ftn_double_complex* BETA,
                    ftn_double_complex* C,
                    ftn_int*            LDC,
                    ftn_len             TRANSA_len,
                    ftn_len             TRANSB_len);

void FORTRAN(ssymm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_single*         ALPHA,
                    ftn_single*         A,
                    ftn_int*            LDA,
                    ftn_single*         B,
                    ftn_int*            LDB,
                    ftn_single*         BETA,
                    ftn_single*         C,
                    ftn_int*            LDC,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len);

void FORTRAN(dsymm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_double*         ALPHA,
                    ftn_double*         A,
                    ftn_int*            LDA,
                    ftn_double*         B,
                    ftn_int*            LDB,
                    ftn_double*         BETA,
                    ftn_double*         C,
                    ftn_int*            LDC,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len);

void FORTRAN(chemm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_complex*        ALPHA,
                    ftn_complex*        A,
                    ftn_int*            LDA,
                    ftn_complex*        B,
                    ftn_int*            LDB,
                    ftn_complex*        BETA,
                    ftn_complex*        C,
                    ftn_int*            LDC,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len);

void FORTRAN(zhemm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_double_complex* ALPHA,
                    ftn_double_complex* A,
                    ftn_int*            LDA,
                    ftn_double_complex* B,
                    ftn_int*            LDB,
                    ftn_double_complex* BETA,
                    ftn_double_complex* C,
                    ftn_int*            LDC,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len);

void FORTRAN(strmm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_char            TRANSA,
                    ftn_char            DIAG,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_single*         ALPHA,
                    ftn_single*         A,
                    ftn_int*            LDA,
                    ftn_single*         B,
                    ftn_int*            LDB,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len,
                    ftn_len             TRANSA_len,
                    ftn_len             DIAG_len);

void FORTRAN(dtrmm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_char            TRANSA,
                    ftn_char            DIAG,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_double*         ALPHA,
                    ftn_double*         A,
                    ftn_int*            LDA,
                    ftn_double*         B,
                    ftn_int*            LDB,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len,
                    ftn_len             TRANSA_len,
                    ftn_len             DIAG_len);

void FORTRAN(ctrmm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_char            TRANSA,
                    ftn_char            DIAG,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_complex*        ALPHA,
                    ftn_complex*        A,
                    ftn_int*            LDA,
                    ftn_complex*        B,
                    ftn_int*            LDB,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len,
                    ftn_len             TRANSA_len,
                    ftn_len             DIAG_len);

void FORTRAN(ztrmm)(ftn_char            SIDE,
                    ftn_char            UPLO,
                    ftn_char            TRANSA,
                    ftn_char            DIAG,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_double_complex* ALPHA,
                    ftn_double_complex* A,
                    ftn_int*            LDA,
                    ftn_double_complex* B,
                    ftn_int*            LDB,
                    ftn_len             SIDE_len,
                    ftn_len             UPLO_len,
                    ftn_len             TRANSA_len,
                    ftn_len             DIAG_len);

void FORTRAN(sgemv)(ftn_char            TRANS,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_single*         ALPHA,
                    ftn_single*         A,
                    ftn_int*            LDA,
                    ftn_single*         X,
                    ftn_int*            INCX,
                    ftn_single*         BETA,
                    ftn_single*         Y,
                    ftn_int*            INCY,
                    ftn_len             TRANS_len);

void FORTRAN(dgemv)(ftn_char            TRANS,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_double*         ALPHA,
                    ftn_double*         A,
                    ftn_int*            LDA,
                    ftn_double*         X,
                    ftn_int*            INCX,
                    ftn_double*         BETA,
                    ftn_double*         Y,
                    ftn_int*            INCY,
                    ftn_len             TRANS_len);

void FORTRAN(cgemv)(ftn_char            TRANS,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_complex*        ALPHA,
                    ftn_complex*        A,
                    ftn_int*            LDA,
                    ftn_complex*        X,
                    ftn_int*            INCX,
                    ftn_complex*        BETA,
                    ftn_complex*        Y,
                    ftn_int*            INCY,
                    ftn_len             TRANS_len);

void FORTRAN(zgemv)(ftn_char            TRANS,
                    ftn_int*            M,
                    ftn_int*            N,
                    ftn_double_complex* ALPHA,
                    ftn_double_complex* A,
                    ftn_int*            LDA,
                    ftn_double_complex* X,
                    ftn_int*            INCX,
                    ftn_double_complex* BETA,
                    ftn_double_complex* Y,
                    ftn_int*            INCY,
                    ftn_len             TRANS_len);

void FORTRAN(sger)(ftn_int*            M,
                   ftn_int*            N,
                   ftn_single*         ALPHA,
                   ftn_single*         X,
                   ftn_int*            INCX,
                   ftn_single*         Y,
                   ftn_int*            INCY,
                   ftn_single*         A,
                   ftn_int*            LDA);

void FORTRAN(dger)(ftn_int*            M,
                   ftn_int*            N,
                   ftn_double*         ALPHA,
                   ftn_double*         X,
                   ftn_int*            INCX,
                   ftn_double*         Y,
                   ftn_int*            INCY,
                   ftn_double*         A,
                   ftn_int*            LDA);

void FORTRAN(cgeru)(ftn_int*            M,
                    ftn_int*            N,
                    ftn_complex*        ALPHA,
                    ftn_complex*        X,
                    ftn_int*            INCX,
                    ftn_complex*        Y,
                    ftn_int*            INCY,
                    ftn_complex*        A,
                    ftn_int*            LDA);

void FORTRAN(cgerc)(ftn_int*            M,
                    ftn_int*            N,
                    ftn_complex*        ALPHA,
                    ftn_complex*        X,
                    ftn_int*            INCX,
                    ftn_complex*        Y,
                    ftn_int*            INCY,
                    ftn_complex*        A,
                    ftn_int*            LDA);

void FORTRAN(zgeru)(ftn_int*            M,
                    ftn_int*            N,
                    ftn_double_complex* ALPHA,
                    ftn_double_complex* X,
                    ftn_int*            INCX,
                    ftn_double_complex* Y,
                    ftn_int*            INCY,
                    ftn_double_complex* A,
                    ftn_int*            LDA);

void FORTRAN(zgerc)(ftn_int*            M,
                    ftn_int*            N,
                    ftn_double_complex* ALPHA,
                    ftn_double_complex* X,
                    ftn_int*            INCX,
                    ftn_double_complex* Y,
                    ftn_int*            INCY,
                    ftn_double_complex* A,
                    ftn_int*            LDA);

void FORTRAN(ssytrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_single*         WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(dsytrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_double*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_double*         WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(dsytrs)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_int*            NRHS,
                     ftn_double*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_double*         B,
                     ftn_int*            LDB,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(chetrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_complex*        A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_complex*        WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(zhetrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_double_complex* A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_double_complex* WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(sgetrf)(ftn_int*            M,
                     ftn_int*            N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_int*            INFO);

void FORTRAN(dgetrf)(ftn_int*            M,
                     ftn_int*            N,
                     ftn_double*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_int*            INFO);

void FORTRAN(cgetrf)(ftn_int*            M,
                     ftn_int*            N,
                     ftn_complex*        A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_int*            INFO);

void FORTRAN(zgetrf)(ftn_int*            M,
                     ftn_int*            N,
                     ftn_double_complex* A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_int*            INFO);

void FORTRAN(zgetrs)(ftn_char UPLO,
                     ftn_int* n,
                     ftn_int* nrhs,
                     ftn_double_complex* A,
                     ftn_int* lda,
                     ftn_int* ipiv,
                     ftn_double_complex* B,
                     ftn_int* ldb,
                     ftn_int* INFO);

void FORTRAN(spotrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(dpotrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_double*         A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(cpotrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_complex*        A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(zpotrf)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_double_complex* A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(ssytri)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_single*         WORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(dsytri)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_double*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_double*         WORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(chetri)(ftn_char            UPLO,
                     ftn_single*         N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_single*         WORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(zhetri)(ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_double_complex* A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_double_complex* WORK,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len);

void FORTRAN(sgetri)(ftn_int*            N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_single*         WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO);

void FORTRAN(dgetri)(ftn_int*            N,
                     ftn_double*         A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_double*         WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO);

void FORTRAN(cgetri)(ftn_int*            N,
                     ftn_complex*        A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_complex*        WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO);

void FORTRAN(zgetri)(ftn_int*            N,
                     ftn_double_complex* A,
                     ftn_int*            LDA,
                     ftn_int*            IPIV,
                     ftn_double_complex* WORK,
                     ftn_int*            LWORK,
                     ftn_int*            INFO);

void FORTRAN(sgesv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_single*         A,
                    ftn_int*            LDA,
                    ftn_int*            IPIV,
                    ftn_single*         B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(dgesv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_double*         A,
                    ftn_int*            LDA,
                    ftn_int*            IPIV,
                    ftn_double*         B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(cgesv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_complex*        A,
                    ftn_int*            LDA,
                    ftn_int*            IPIV,
                    ftn_complex*        B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(zgesv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_double_complex* A,
                    ftn_int*            LDA,
                    ftn_int*            IPIV,
                    ftn_double_complex* B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(sgtsv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_single*         DL,
                    ftn_single*         D,
                    ftn_single*         DU,
                    ftn_single*         B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(dgtsv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_double*         DL,
                    ftn_double*         D,
                    ftn_double*         DU,
                    ftn_double*         B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(cgtsv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_complex*        DL,
                    ftn_complex*        D,
                    ftn_complex*        DU,
                    ftn_complex*        B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(zgtsv)(ftn_int*            N,
                    ftn_int*            NRHS,
                    ftn_double_complex* DL,
                    ftn_double_complex* D,
                    ftn_double_complex* DU,
                    ftn_double_complex* B,
                    ftn_int*            LDB,
                    ftn_int*            INFO);

void FORTRAN(strtri)(ftn_char            UPLO,
                     ftn_char            DIAG,
                     ftn_int*            N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len,
                     ftn_len             DIAG_len);

void FORTRAN(dtrtri)(ftn_char            UPLO,
                     ftn_char            DIAG,
                     ftn_int*            N,
                     ftn_double*         A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len,
                     ftn_len             DIAG_len);

void FORTRAN(ctrtri)(ftn_char            UPLO,
                     ftn_char            DIAG,
                     ftn_int*            N,
                     ftn_complex*        A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len,
                     ftn_len             DIAG_len);

void FORTRAN(ztrtri)(ftn_char            UPLO,
                     ftn_char            DIAG,
                     ftn_int*            N,
                     ftn_double_complex* A,
                     ftn_int*            LDA,
                     ftn_int*            INFO,
                     ftn_len             UPLO_len,
                     ftn_len             DIAG_len);

void FORTRAN(ssygvx)(ftn_int*            ITYPE,
                     ftn_char            JOBZ,
                     ftn_char            RANGE,
                     ftn_char            UPLO,
                     ftn_int*            N,
                     ftn_single*         A,
                     ftn_int*            LDA,
                     ftn_single*         B,
                     ftn_int*            LDB,
                     ftn_single*         VL,
                     ftn_single*         VU,
                     ftn_int*            IL,
                     ftn_int*            IU,
                     ftn_single*         ABSTOL,
                     ftn_int*            M,
                     ftn_single*         W,
                     ftn_single*         Z,
                     ftn_int*LDZ,
                     ftn_single* WORK, ftn_int* LWORK, ftn_int* IWORK, ftn_int* IFAIL, ftn_int* INFO, ftn_len JOBZ_len,
                     ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(dsygvx)(ftn_int* ITYPE, ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_double* A,
                     ftn_int* LDA, ftn_double* B, ftn_int* LDB, ftn_double* VL, ftn_double* VU, ftn_int* IL,
                     ftn_int* IU, ftn_double* ABSTOL, ftn_int* M, ftn_double* W, ftn_double* Z, ftn_int* LDZ,
                     ftn_double* WORK, ftn_int* LWORK, ftn_int* IWORK, ftn_int* IFAIL, ftn_int* INFO, ftn_len JOBZ_len,
                     ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(chegvx)(ftn_int* ITYPE, ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_complex* A,
                     ftn_int* LDA, ftn_complex* B, ftn_int* LDB, ftn_single* VL, ftn_single* VU, ftn_int* IL,
                     ftn_int* IU, ftn_single* ABSTOL, ftn_int* M, ftn_single* W, ftn_complex* Z, ftn_int* LDZ,
                     ftn_complex* WORK, ftn_int* LWORK, ftn_single* RWORK, ftn_int* IWORK, ftn_int* IFAIL,
                     ftn_int* INFO, ftn_len JOBZ_len, ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(zhegvx)(ftn_int* ITYPE, ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_double_complex* A,
                     ftn_int* LDA, ftn_double_complex* B, ftn_int* LDB, ftn_double* VL, ftn_double* VU, ftn_int* IL,
                     ftn_int* IU, ftn_double* ABSTOL, ftn_int* M, ftn_double* W, ftn_double_complex* Z, ftn_int* LDZ,
                     ftn_double_complex* WORK, ftn_int* LWORK, ftn_double* RWORK, ftn_int* IWORK, ftn_int* IFAIL,
                     ftn_int* INFO, ftn_len JOBZ_len, ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(ssyev)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_single* A, ftn_int* LDA, ftn_single* W,
                    ftn_single* WORK, ftn_int* LWORK, ftn_int* INFO, ftn_len JOBZ_len, ftn_len UPLO_len);

void FORTRAN(dsyev)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_double* A, ftn_int* LDA, ftn_double* W,
                    ftn_double* WORK, ftn_int* LWORK, ftn_int* INFO, ftn_len JOBZ_len, ftn_len UPLO_len);

void FORTRAN(cheev)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_complex* A, ftn_int* LDA, ftn_single* W,
                    ftn_complex* WORK, ftn_int* LWORK, ftn_single* RWORK, ftn_int* INFO, ftn_len JOBZ_len,
                    ftn_len UPLO_len);

void FORTRAN(zheev)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_double_complex* A, ftn_int* LDA, ftn_double* W,
                    ftn_double_complex* WORK, ftn_int* LWORK, ftn_double* RWORK, ftn_int* INFO, ftn_len JOBZ_len,
                    ftn_len UPLO_len);

void FORTRAN(ssyevd)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_single* A, ftn_int* LDA, ftn_single* W,
                     ftn_single* WORK, ftn_int* LWORK, ftn_int* IWORK, ftn_int* LIWORK, ftn_int* INFO, ftn_len JOBZ_len,
                     ftn_len UPLO_len);

void FORTRAN(dsyevd)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_double* A, ftn_int* LDA, ftn_double* W,
                     ftn_double* WORK, ftn_int* LWORK, ftn_int* IWORK, ftn_int* LIWORK, ftn_int* INFO, ftn_len JOBZ_len,
                     ftn_len UPLO_len);

void FORTRAN(cheevd)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_complex* A, ftn_int* LDA, ftn_single* W,
                     ftn_complex* WORK, ftn_int* LWORK, ftn_single* RWORK, ftn_int* LRWORK, ftn_int* IWORK,
                     ftn_int* LIWORK, ftn_int* INFO, ftn_len JOBZ_len, ftn_len UPLO_len);

void FORTRAN(zheevd)(ftn_char JOBZ, ftn_char UPLO, ftn_int* N, ftn_double_complex* A, ftn_int* LDA, ftn_double* W,
                     ftn_double_complex* WORK, ftn_int* LWORK, ftn_double* RWORK, ftn_int* LRWORK, ftn_int* IWORK,
                     ftn_int* LIWORK, ftn_int* INFO, ftn_len JOBZ_len, ftn_len UPLO_len);

void FORTRAN(ssyevx)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_single* A, ftn_int* LDA,
                     ftn_single* VL, ftn_single* VU, ftn_int* IL, ftn_int* IU, ftn_single* ABSTOL, ftn_int* M,
                     ftn_single* W, ftn_single* Z, ftn_int* LDZ, ftn_single* WORK, ftn_int* LWORK, ftn_int* IWORK,
                     ftn_int* IFAIL, ftn_int* INFO, ftn_len JOBZ_len, ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(dsyevx)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_double* A, ftn_int* LDA,
                     ftn_double* VL, ftn_double* VU, ftn_int* IL, ftn_int* IU, ftn_double* ABSTOL, ftn_int* M,
                     ftn_double* W, ftn_double* Z, ftn_int* LDZ, ftn_double* WORK, ftn_int* LWORK, ftn_int* IWORK,
                     ftn_int* IFAIL, ftn_int* INFO, ftn_len JOBZ_len, ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(cheevx)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_complex* A, ftn_int* LDA,
                     ftn_single* VL, ftn_single* VU, ftn_int* IL, ftn_int* IU, ftn_single* ABSTOL, ftn_int* M,
                     ftn_single* W, ftn_complex* Z, ftn_int* LDZ, ftn_complex* WORK, ftn_int* LWORK, ftn_single* RWORK,
                     ftn_int* IWORK, ftn_int* IFAIL, ftn_int* INFO, ftn_len JOBZ_len, ftn_len RANGE_len,
                     ftn_len UPLO_len);

void FORTRAN(zheevx)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_double_complex* A, ftn_int* LDA,
                     ftn_double* VL, ftn_double* VU, ftn_int* IL, ftn_int* IU, ftn_double* ABSTOL, ftn_int* M,
                     ftn_double* W, ftn_double_complex* Z, ftn_int* LDZ, ftn_double_complex* WORK, ftn_int* LWORK,
                     ftn_double* RWORK, ftn_int* IWORK, ftn_int* IFAIL, ftn_int* INFO, ftn_len JOBZ_len,
                     ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(ssyevr)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_single* A, ftn_int* LDA,
                     ftn_single* VL, ftn_single* VU, ftn_int* IL, ftn_int* IU, ftn_single* ABSTOL, ftn_int* M,
                     ftn_single* W, ftn_single* Z, ftn_int* LDZ, ftn_int* ISUPPZ, ftn_single* WORK, ftn_int* LWORK,
                     ftn_int* IWORK, ftn_int* LIWORK, ftn_int* INFO, ftn_len JOBZ_len, ftn_len RANGE_len,
                     ftn_len UPLO_len);

void FORTRAN(dsyevr)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_double* A, ftn_int* LDA,
                     ftn_double* VL, ftn_double* VU, ftn_int* IL, ftn_int* IU, ftn_double* ABSTOL, ftn_int* M,
                     ftn_double* W, ftn_double* Z, ftn_int* LDZ, ftn_int* ISUPPZ, ftn_double* WORK, ftn_int* LWORK,
                     ftn_int* IWORK, ftn_int* LIWORK, ftn_int* INFO, ftn_len JOBZ_len, ftn_len RANGE_len,
                     ftn_len UPLO_len);

void FORTRAN(cheevr)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_single* A, ftn_int* LDA,
                     ftn_single* VL, ftn_single* VU, ftn_int* IL, ftn_int* IU, ftn_single* ABSTOL, ftn_int* M,
                     ftn_single* W, ftn_complex* Z, ftn_int* LDZ, ftn_int* ISUPPZ, ftn_complex* WORK, ftn_int* LWORK,
                     ftn_single* RWORK, ftn_int* LRWORK, ftn_int* IWORK, ftn_int* LIWORK, ftn_int* INFO,
                     ftn_len JOBZ_len, ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(zheevr)(ftn_char JOBZ, ftn_char RANGE, ftn_char UPLO, ftn_int* N, ftn_double_complex* A, ftn_int* LDA,
                     ftn_double* VL, ftn_double* VU, ftn_int* IL, ftn_int* IU, ftn_double* ABSTOL, ftn_int* M,
                     ftn_double* W, ftn_double_complex* Z, ftn_int* LDZ, ftn_int* ISUPPZ, ftn_double_complex* WORK,
                     ftn_int* LWORK, ftn_double* RWORK, ftn_int* LRWORK, ftn_int* IWORK, ftn_int* LIWORK, ftn_int* INFO,
                     ftn_len JOBZ_len, ftn_len RANGE_len, ftn_len UPLO_len);

void FORTRAN(sgeqrf)(ftn_int* M, ftn_int* N, ftn_single* A, ftn_int* LDA, ftn_single* TAU, ftn_single* WORK,
                     ftn_int* LWORK, ftn_int* INFO);

void FORTRAN(dgeqrf)(ftn_int* M, ftn_int* N, ftn_double* A, ftn_int* LDA, ftn_double* TAU, ftn_double* WORK,
                     ftn_int* LWORK, ftn_int* INFO);

void FORTRAN(cgeqrf)(ftn_int* M, ftn_int* N, ftn_complex* A, ftn_int* LDA, ftn_complex* TAU, ftn_complex* WORK,
                     ftn_int* LWORK, ftn_int* INFO);

void FORTRAN(zgeqrf)(ftn_int* M, ftn_int* N, ftn_double_complex* A, ftn_int* LDA, ftn_double_complex* TAU,
                     ftn_double_complex* WORK, ftn_int* LWORK, ftn_int* INFO);

void FORTRAN(sscal)(const ftn_int* N, const ftn_single* ALPHA, ftn_single* X, const ftn_int* INCX);

void FORTRAN(dscal)(const ftn_int* N, const ftn_double* ALPHA, ftn_double* X, const ftn_int* INCX);

void FORTRAN(cscal)(const ftn_int* N, const ftn_complex* ALPHA, ftn_complex* X, const ftn_int* INCX);

void FORTRAN(zscal)(const ftn_int* N, const ftn_double_complex* ALPHA, ftn_double_complex* X, const ftn_int* INCX);

void FORTRAN(dlartg)(ftn_double* f, ftn_double* g, ftn_double* cs, ftn_double* sn, ftn_double* r);
void FORTRAN(daxpy)(const ftn_int*, const ftn_double*, const ftn_double*, const ftn_int*, ftn_double*, ftn_int*);
void FORTRAN(zaxpy)(const ftn_int*, const ftn_double_complex*, const ftn_double_complex*, const ftn_int*,
                    ftn_double_complex*, ftn_int*);

void FORTRAN(zgesvd)(ftn_char jobu, ftn_char jobvt, const ftn_int* m, const ftn_int* n, const ftn_double_complex* A,
                     const ftn_int* lda, const ftn_double* S, ftn_double_complex* U, const ftn_int* ldu,
                     ftn_double_complex* Vt, const ftn_int* ldvt, ftn_double_complex* work, const ftn_int* lwork,
                     ftn_double* rwork, ftn_int* info);
}

#endif
