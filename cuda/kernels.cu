#include "kernels.cuh"
#include "cuda_utils.h"
#include <cfloat>

/* ────────────────────────── elementwise 1D ────────────────────────── */

__global__ void k_relu_forward(const float *Z, float *A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = Z[i] > 0.f ? Z[i] : 0.f;
}

__global__ void k_relu_backward(const float *Z, const float *dA, float *dZ, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dZ[i] = Z[i] > 0.f ? dA[i] : 0.f;
}

__global__ void k_axpy(float *A, float alpha, const float *B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] += alpha * B[i];
}

__global__ void k_sub_scale(const float *A, const float *B, float scale, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = (A[i] - B[i]) * scale;
}

/* ───────────────────────────── bias add ───────────────────────────── */

__global__ void k_add_bias(float *A, const float *b, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) A[i*cols + j] += b[j];
}

/* ─────────────────────── column sum (small rows) ──────────────────── */

__global__ void k_col_sum(const float *A, float *out, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.f;
    for (int i = 0; i < rows; i++) s += A[i*cols + j];
    out[j] = s;
}

/* ───────────────────── stable row-wise softmax ────────────────────── */
/* One warp per row; cols must be ≤ 32 (our output layer: 10). */

__global__ void k_softmax_forward(const float *Z, float *A, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;           /* 0..31 */
    if (row >= rows) return;
    const float *z = Z + row * cols;
    float       *a = A + row * cols;

    float local_max = -FLT_MAX;
    for (int j = tid; j < cols; j += 32)
        local_max = fmaxf(local_max, z[j]);
    for (int o = 16; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, o));

    float local_sum = 0.f;
    for (int j = tid; j < cols; j += 32) {
        float e = __expf(z[j] - local_max);
        a[j] = e;
        local_sum += e;
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, o);

    for (int j = tid; j < cols; j += 32)
        a[j] /= local_sum;
}

/* ────────────────────────── cross-entropy ─────────────────────────── */

__global__ void k_ce_loss(const float *A, const float *Y, float *loss_sum,
                          int rows, int cols, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float *a = A + i*cols;
    const float *y = Y + i*cols;
    float s = 0.f;
    for (int j = 0; j < cols; j++)
        if (y[j] > 0.f) s -= logf(a[j] + eps);
    atomicAdd(loss_sum, s);
}

/* ───────────────────────────── gather ─────────────────────────────── */

__global__ void k_gather_rows(const float *src, const int *idx, float *dst,
                              int bs, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (b < bs && j < cols)
        dst[b*cols + j] = src[idx[b]*cols + j];
}

/* ───────────────────────── argmax + count ─────────────────────────── */

__global__ void k_argmax_count(const float *A, const int *labels, int *correct,
                               int rows, int cols, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const float *a = A + i*cols;
    int pred = 0;
    float mx = a[0];
    for (int j = 1; j < cols; j++)
        if (a[j] > mx) { mx = a[j]; pred = j; }
    if (pred == labels[offset + i]) atomicAdd(correct, 1);
}

/* ──────────────────────────── launchers ───────────────────────────── */

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void launch_relu_forward(const float *Z, float *A, int n) {
    int bs = 256;
    k_relu_forward<<<ceil_div(n, bs), bs>>>(Z, A, n);
}

void launch_relu_backward(const float *Z, const float *dA, float *dZ, int n) {
    int bs = 256;
    k_relu_backward<<<ceil_div(n, bs), bs>>>(Z, dA, dZ, n);
}

void launch_softmax_forward(const float *Z, float *A, int rows, int cols) {
    k_softmax_forward<<<rows, 32>>>(Z, A, rows, cols);
}

void launch_add_bias(float *A, const float *b, int rows, int cols) {
    dim3 block(32, 8);
    dim3 grid(ceil_div(cols, block.x), ceil_div(rows, block.y));
    k_add_bias<<<grid, block>>>(A, b, rows, cols);
}

void launch_col_sum(const float *A, float *out, int rows, int cols) {
    int bs = 128;
    k_col_sum<<<ceil_div(cols, bs), bs>>>(A, out, rows, cols);
}

void launch_axpy(float *A, float alpha, const float *B, int n) {
    int bs = 256;
    k_axpy<<<ceil_div(n, bs), bs>>>(A, alpha, B, n);
}

void launch_sub_scale(const float *A, const float *B, float scale, float *C, int n) {
    int bs = 256;
    k_sub_scale<<<ceil_div(n, bs), bs>>>(A, B, scale, C, n);
}

void launch_ce_loss(const float *A, const float *Y, float *loss_sum,
                    int rows, int cols, float eps) {
    int bs = 128;
    k_ce_loss<<<ceil_div(rows, bs), bs>>>(A, Y, loss_sum, rows, cols, eps);
}

void launch_gather_rows(const float *src, const int *idx, float *dst,
                        int bs, int cols) {
    int tx = 128;
    dim3 grid(ceil_div(cols, tx), bs);
    k_gather_rows<<<grid, tx>>>(src, idx, dst, bs, cols);
}

void launch_argmax_count(const float *A, const int *labels, int *correct,
                         int rows, int cols, int offset) {
    int bs = 128;
    k_argmax_count<<<ceil_div(rows, bs), bs>>>(A, labels, correct, rows, cols, offset);
}
