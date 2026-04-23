#pragma once
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct {
    int rows, cols;
    double *data;  /* row-major */
} Matrix;

#define MAT_AT(m, i, j) ((m)->data[(i)*(m)->cols+(j)])

Matrix mat_create(int rows, int cols);
void   mat_free(Matrix *m);
void   mat_zero(Matrix *m);
void   mat_copy(const Matrix *src, Matrix *dst);

/* C = A * B  — A:(M,K)  B:(K,N)  C:(M,N) */
void mat_mul(const Matrix *A, const Matrix *B, Matrix *C);
/* C = A^T * B  — A:(K,M)  B:(K,N)  C:(M,N) */
void mat_mul_ta(const Matrix *A, const Matrix *B, Matrix *C);
/* C = A * B^T  — A:(M,K)  B:(N,K)  C:(M,N) */
void mat_mul_tb(const Matrix *A, const Matrix *B, Matrix *C);

/* element-wise in-place: A += alpha*B */
void mat_axpy(Matrix *A, double alpha, const Matrix *B);
/* element-wise: C = A - B */
void mat_sub(const Matrix *A, const Matrix *B, Matrix *C);
/* element-wise in-place: A *= s */
void mat_scale(Matrix *A, double s);
/* element-wise: C = A .* B */
void mat_hadamard(const Matrix *A, const Matrix *B, Matrix *C);
/* broadcast: add bias row b (1,n) to every row of A (m,n), in-place */
void mat_add_bias(Matrix *A, const Matrix *b);
/* column sums: out(1,cols) = sum of rows of A */
void mat_col_sum(const Matrix *A, Matrix *out);

/* load / save raw binary (row-major doubles) */
Matrix mat_load_bin(const char *path, int rows, int cols);
void   mat_save_bin(const char *path, const Matrix *m);
