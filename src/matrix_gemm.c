/* Naive (triple-loop) GEMM implementations.
 *
 * Shared by the naive and OpenMP builds — the OpenMP pragmas below are
 * no-ops unless the file is compiled with -fopenmp.
 *
 * The OpenBLAS build replaces this file with blas/matrix_blas.c, which
 * provides cblas_dgemm-based versions of the same three functions.
 */

#include "matrix.h"

/* C = A*B  — i-k-j order is cache-friendly for row-major. */
void mat_mul(const Matrix *A, const Matrix *B, Matrix *C) {
    int M = A->rows, K = A->cols, N = B->cols;
    assert(B->rows == K && C->rows == M && C->cols == N);
    mat_zero(C);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            double a = A->data[i*K + k];
            for (int j = 0; j < N; j++)
                C->data[i*N + j] += a * B->data[k*N + j];
        }
}

/* C = A^T * B  — A:(K,M)  B:(K,N)  C:(M,N)
   C[i][j] = sum_k A[k][i] * B[k][j] */
void mat_mul_ta(const Matrix *A, const Matrix *B, Matrix *C) {
    int K = A->rows, M = A->cols, N = B->cols;
    assert(B->rows == K && C->rows == M && C->cols == N);
    mat_zero(C);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            double a = A->data[k*M + i];  /* A[k][i] = A^T[i][k] */
            for (int j = 0; j < N; j++)
                C->data[i*N + j] += a * B->data[k*N + j];
        }
}

/* C = A * B^T  — A:(M,K)  B:(N,K)  C:(M,N)
   C[i][j] = sum_k A[i][k] * B[j][k] */
void mat_mul_tb(const Matrix *A, const Matrix *B, Matrix *C) {
    int M = A->rows, K = A->cols, N = B->rows;
    assert(B->cols == K && C->rows == M && C->cols == N);
    mat_zero(C);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            double a = A->data[i*K + k];
            for (int j = 0; j < N; j++)
                C->data[i*N + j] += a * B->data[j*K + k];
        }
}
