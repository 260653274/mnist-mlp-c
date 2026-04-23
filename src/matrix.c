#include "matrix.h"
#include <stdio.h>
#include <math.h>

Matrix mat_create(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double *)calloc((size_t)rows * cols, sizeof(double));
    assert(m.data != NULL);
    return m;
}

void mat_free(Matrix *m) {
    free(m->data);
    m->data = NULL;
    m->rows = m->cols = 0;
}

void mat_zero(Matrix *m) {
    memset(m->data, 0, (size_t)m->rows * m->cols * sizeof(double));
}

void mat_copy(const Matrix *src, Matrix *dst) {
    assert(src->rows == dst->rows && src->cols == dst->cols);
    memcpy(dst->data, src->data, (size_t)src->rows * src->cols * sizeof(double));
}

/* GEMM implementations live in:
 *   src/matrix_gemm.c  — naive (triple-loop, OMP-aware)
 *   blas/matrix_blas.c — OpenBLAS (cblas_dgemm)
 * The build system picks exactly one of the two per target. */

void mat_axpy(Matrix *A, double alpha, const Matrix *B) {
    assert(A->rows == B->rows && A->cols == B->cols);
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] += alpha * B->data[i];
}

void mat_sub(const Matrix *A, const Matrix *B, Matrix *C) {
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(C->rows == A->rows && C->cols == A->cols);
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        C->data[i] = A->data[i] - B->data[i];
}

void mat_scale(Matrix *A, double s) {
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] *= s;
}

void mat_hadamard(const Matrix *A, const Matrix *B, Matrix *C) {
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(C->rows == A->rows && C->cols == A->cols);
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        C->data[i] = A->data[i] * B->data[i];
}

void mat_add_bias(Matrix *A, const Matrix *b) {
    assert(b->rows == 1 && b->cols == A->cols);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            A->data[i*A->cols + j] += b->data[j];
}

void mat_col_sum(const Matrix *A, Matrix *out) {
    assert(out->rows == 1 && out->cols == A->cols);
    mat_zero(out);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            out->data[j] += A->data[i*A->cols + j];
}

Matrix mat_load_bin(const char *path, int rows, int cols) {
    Matrix m = mat_create(rows, cols);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t n = (size_t)rows * cols;
    size_t r = fread(m.data, sizeof(double), n, f);
    if (r != n) { fprintf(stderr, "short read in %s\n", path); exit(1); }
    fclose(f);
    return m;
}

void mat_save_bin(const char *path, const Matrix *m) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite(m->data, sizeof(double), (size_t)m->rows * m->cols, f);
    fclose(f);
}
