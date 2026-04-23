#pragma once
#include <cublas_v2.h>

/* Network parameters on GPU (float32, row-major). */
typedef struct {
    float *W1, *b1;   /* (784,H)  (1,H)  */
    float *W2, *b2;   /* (H,10)   (1,10) */
    int hidden;
} CudaParams;

/* Per-batch workspace on GPU. */
typedef struct {
    int batch_size, hidden;
    float *X, *Y;                   /* (bs,784)  (bs,10) */
    float *Z1, *A1, *Z2, *A2;       /* forward */
    float *dZ2, *dA1, *dZ1;         /* backward */
    float *dW1, *db1, *dW2, *db2;   /* gradients */
} CudaWorkSpace;

/* Forward-only workspace (for evaluation). */
typedef struct {
    int rows, hidden;
    float *Z1, *A1, *Z2, *A2;
} CudaEvalWS;

CudaParams     cuda_params_create(int hidden);
void           cuda_params_free(CudaParams *p);

CudaWorkSpace  cuda_ws_create(int batch_size, int hidden);
void           cuda_ws_free(CudaWorkSpace *ws);

CudaEvalWS     cuda_evalws_create(int rows, int hidden);
void           cuda_evalws_free(CudaEvalWS *e);

/* Upload float32 host buffer to device. */
void cuda_upload(float *d_dst, const float *h_src, size_t n);
/* Download device buffer to float32 host. */
void cuda_download(float *h_dst, const float *d_src, size_t n);

/* Training pipeline (uses ws->X, ws->Y already populated on device). */
void cuda_forward (cublasHandle_t h, const CudaParams *p, CudaWorkSpace *ws);
void cuda_backward(cublasHandle_t h, const CudaParams *p, CudaWorkSpace *ws);
void cuda_update  (const CudaParams *p, const CudaWorkSpace *ws, float lr);

/* Eval forward: X (rows,784) on device → fills e->A2 (rows,10). */
void cuda_eval_forward(cublasHandle_t h, const CudaParams *p,
                       const float *X, CudaEvalWS *e, int rows);
