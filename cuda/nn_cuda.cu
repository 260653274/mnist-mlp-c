#include "nn_cuda.cuh"
#include "kernels.cuh"
#include "cuda_utils.h"

/* ───────────────────────── allocate / free ────────────────────────── */

static float *d_alloc_zero(size_t n) {
    float *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(float)));
    return p;
}

CudaParams cuda_params_create(int hidden) {
    CudaParams p;
    p.hidden = hidden;
    p.W1 = d_alloc_zero((size_t)784   * hidden);
    p.b1 = d_alloc_zero((size_t)hidden);
    p.W2 = d_alloc_zero((size_t)hidden * 10);
    p.b2 = d_alloc_zero((size_t)10);
    return p;
}

void cuda_params_free(CudaParams *p) {
    cudaFree(p->W1); cudaFree(p->b1);
    cudaFree(p->W2); cudaFree(p->b2);
    p->W1 = p->b1 = p->W2 = p->b2 = nullptr;
}

CudaWorkSpace cuda_ws_create(int bs, int hidden) {
    CudaWorkSpace w;
    w.batch_size = bs;
    w.hidden     = hidden;
    w.X   = d_alloc_zero((size_t)bs * 784);
    w.Y   = d_alloc_zero((size_t)bs * 10);
    w.Z1  = d_alloc_zero((size_t)bs * hidden);
    w.A1  = d_alloc_zero((size_t)bs * hidden);
    w.Z2  = d_alloc_zero((size_t)bs * 10);
    w.A2  = d_alloc_zero((size_t)bs * 10);
    w.dZ2 = d_alloc_zero((size_t)bs * 10);
    w.dA1 = d_alloc_zero((size_t)bs * hidden);
    w.dZ1 = d_alloc_zero((size_t)bs * hidden);
    w.dW1 = d_alloc_zero((size_t)784   * hidden);
    w.db1 = d_alloc_zero((size_t)hidden);
    w.dW2 = d_alloc_zero((size_t)hidden * 10);
    w.db2 = d_alloc_zero((size_t)10);
    return w;
}

void cuda_ws_free(CudaWorkSpace *w) {
    cudaFree(w->X);   cudaFree(w->Y);
    cudaFree(w->Z1);  cudaFree(w->A1);  cudaFree(w->Z2);  cudaFree(w->A2);
    cudaFree(w->dZ2); cudaFree(w->dA1); cudaFree(w->dZ1);
    cudaFree(w->dW1); cudaFree(w->db1); cudaFree(w->dW2); cudaFree(w->db2);
}

CudaEvalWS cuda_evalws_create(int rows, int hidden) {
    CudaEvalWS e;
    e.rows   = rows;
    e.hidden = hidden;
    e.Z1 = d_alloc_zero((size_t)rows * hidden);
    e.A1 = d_alloc_zero((size_t)rows * hidden);
    e.Z2 = d_alloc_zero((size_t)rows * 10);
    e.A2 = d_alloc_zero((size_t)rows * 10);
    return e;
}

void cuda_evalws_free(CudaEvalWS *e) {
    cudaFree(e->Z1); cudaFree(e->A1); cudaFree(e->Z2); cudaFree(e->A2);
}

void cuda_upload(float *d_dst, const float *h_src, size_t n) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n * sizeof(float), cudaMemcpyHostToDevice));
}

void cuda_download(float *h_dst, const float *d_src, size_t n) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToHost));
}

/* ──────────────── row-major GEMM wrappers over cuBLAS ──────────────── *
 * cuBLAS is column-major; we store row-major.  Using the identity
 *     (A·B) stored row-major  ==  (Bᵀ·Aᵀ) stored column-major
 * we swap the two operands and choose transA/transB accordingly.
 */

/* C(M,N) = A(M,K) * B(K,N) */
static inline void gemm_nn(cublasHandle_t h, int M, int N, int K,
                           float alpha, const float *A, const float *B,
                           float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, B, N, A, K,
                             &beta,  C, N));
}

/* C(M,N) = Aᵀ · B,  where A stored row-major as (K,M), B as (K,N). */
static inline void gemm_tn(cublasHandle_t h, int M, int N, int K,
                           float alpha, const float *A, const float *B,
                           float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K,
                             &alpha, B, N, A, M,
                             &beta,  C, N));
}

/* C(M,N) = A · Bᵀ, where A stored row-major (M,K), B as (N,K). */
static inline void gemm_nt(cublasHandle_t h, int M, int N, int K,
                           float alpha, const float *A, const float *B,
                           float beta, float *C) {
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, B, K, A, K,
                             &beta,  C, N));
}

/* ─────────────────────────── forward ──────────────────────────────── */

void cuda_forward(cublasHandle_t h, const CudaParams *p, CudaWorkSpace *ws) {
    int bs = ws->batch_size, H = ws->hidden;

    /* Z1 = X · W1 */
    gemm_nn(h, bs, H, 784, 1.f, ws->X, p->W1, 0.f, ws->Z1);
    launch_add_bias(ws->Z1, p->b1, bs, H);
    launch_relu_forward(ws->Z1, ws->A1, bs * H);

    /* Z2 = A1 · W2 */
    gemm_nn(h, bs, 10, H, 1.f, ws->A1, p->W2, 0.f, ws->Z2);
    launch_add_bias(ws->Z2, p->b2, bs, 10);
    launch_softmax_forward(ws->Z2, ws->A2, bs, 10);
}

/* ─────────────────────────── backward ─────────────────────────────── */

void cuda_backward(cublasHandle_t h, const CudaParams *p, CudaWorkSpace *ws) {
    int bs = ws->batch_size, H = ws->hidden;

    /* dZ2 = (A2 - Y) / bs */
    launch_sub_scale(ws->A2, ws->Y, 1.f / (float)bs, ws->dZ2, bs * 10);

    /* dW2 = A1ᵀ · dZ2   (H,10) */
    gemm_tn(h, H, 10, bs, 1.f, ws->A1, ws->dZ2, 0.f, ws->dW2);
    /* db2 = colsum(dZ2) */
    launch_col_sum(ws->dZ2, ws->db2, bs, 10);

    /* dA1 = dZ2 · W2ᵀ   (bs,H) */
    gemm_nt(h, bs, H, 10, 1.f, ws->dZ2, p->W2, 0.f, ws->dA1);
    launch_relu_backward(ws->Z1, ws->dA1, ws->dZ1, bs * H);

    /* dW1 = Xᵀ · dZ1   (784,H) */
    gemm_tn(h, 784, H, bs, 1.f, ws->X, ws->dZ1, 0.f, ws->dW1);
    launch_col_sum(ws->dZ1, ws->db1, bs, H);
}

/* ───────────────────── SGD parameter update ───────────────────────── */

void cuda_update(const CudaParams *p, const CudaWorkSpace *ws, float lr) {
    int H = p->hidden;
    launch_axpy(p->W1, -lr, ws->dW1, 784 * H);
    launch_axpy(p->b1, -lr, ws->db1, H);
    launch_axpy(p->W2, -lr, ws->dW2, H * 10);
    launch_axpy(p->b2, -lr, ws->db2, 10);
}

/* ─────────────────── forward-only for evaluation ──────────────────── */

void cuda_eval_forward(cublasHandle_t h, const CudaParams *p,
                       const float *X, CudaEvalWS *e, int rows) {
    int H = p->hidden;
    gemm_nn(h, rows, H, 784, 1.f, X, p->W1, 0.f, e->Z1);
    launch_add_bias(e->Z1, p->b1, rows, H);
    launch_relu_forward(e->Z1, e->A1, rows * H);

    gemm_nn(h, rows, 10, H, 1.f, e->A1, p->W2, 0.f, e->Z2);
    launch_add_bias(e->Z2, p->b2, rows, 10);
    launch_softmax_forward(e->Z2, e->A2, rows, 10);
}
