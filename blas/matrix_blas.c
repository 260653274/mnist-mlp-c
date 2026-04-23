/* OpenBLAS (cblas_dgemm) implementations of the three GEMMs declared in
 * ../src/matrix.h.  Same function signatures as src/matrix_gemm.c —
 * the OpenBLAS build links this file instead of matrix_gemm.c.
 */

#include "../src/matrix.h"
#include <cblas.h>

/* C = A * B  — A:(M,K)  B:(K,N)  C:(M,N) */
void mat_mul(const Matrix *A, const Matrix *B, Matrix *C) {
    int M = A->rows, K = A->cols, N = B->cols;
    assert(B->rows == K && C->rows == M && C->cols == N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0,
                A->data, K,
                B->data, N,
                0.0, C->data, N);
}

/* C = A^T * B  — A:(K,M)  B:(K,N)  C:(M,N) */
void mat_mul_ta(const Matrix *A, const Matrix *B, Matrix *C) {
    int K = A->rows, M = A->cols, N = B->cols;
    assert(B->rows == K && C->rows == M && C->cols == N);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, 1.0,
                A->data, M,   /* lda = cols of stored A = M */
                B->data, N,
                0.0, C->data, N);
}

/* C = A * B^T  — A:(M,K)  B:(N,K)  C:(M,N) */
void mat_mul_tb(const Matrix *A, const Matrix *B, Matrix *C) {
    int M = A->rows, K = A->cols, N = B->rows;
    assert(B->cols == K && C->rows == M && C->cols == N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0,
                A->data, K,
                B->data, K,   /* ldb = cols of stored B = K */
                0.0, C->data, N);
}
