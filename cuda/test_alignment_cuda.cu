/* CUDA alignment test:
 *   - load Python init weights + batch-0 inputs (fp64 → fp32 on upload)
 *   - single forward + backward on GPU
 *   - compare every intermediate against Python fp64 reference
 *
 * Tolerance: 1e-4  (fp32 vs fp64; single-precision accumulation error).
 */

#include "nn_cuda.cuh"
#include "kernels.cuh"
#include "cuda_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HIDDEN      512
#define BATCH_SIZE  50
#define TOL         1e-4

#define WEIGHTS_DIR "weights/"
#define LOGS_DIR    "logs/"

/* Load fp64 .bin into a float32 host buffer. */
static float *load_bin_fp32_host(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    double *h64 = (double *)malloc(n * sizeof(double));
    if (fread(h64, sizeof(double), n, f) != n) {
        fprintf(stderr, "short read %s\n", path); exit(1);
    }
    fclose(f);
    float *h32 = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) h32[i] = (float)h64[i];
    free(h64);
    return h32;
}

static void upload_bin_to_device(const char *path, float *d_dst, size_t n) {
    float *h32 = load_bin_fp32_host(path, n);
    CUDA_CHECK(cudaMemcpy(d_dst, h32, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h32);
}

static double max_abs_err_fp64_vs_fp32(const float *got_h32, const double *ref_f64, size_t n) {
    double e = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got_h32[i] - ref_f64[i]);
        if (d > e) e = d;
    }
    return e;
}

static int check(const char *name, const float *d_got, const char *ref_path,
                 size_t n, double tol) {
    float *h_got = (float *)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_got, d_got, n * sizeof(float), cudaMemcpyDeviceToHost));

    FILE *f = fopen(ref_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ref_path); exit(1); }
    double *ref = (double *)malloc(n * sizeof(double));
    if (fread(ref, sizeof(double), n, f) != n) {
        fprintf(stderr, "short read %s\n", ref_path); exit(1);
    }
    fclose(f);

    double e = max_abs_err_fp64_vs_fp32(h_got, ref, n);
    int pass = e < tol;
    printf("  %-12s  max_err=%.3e  %s\n", name, e, pass ? "PASS" : "FAIL ***");

    free(h_got); free(ref);
    return pass;
}

int main(void) {
    cublasHandle_t cub; CUBLAS_CHECK(cublasCreate(&cub));

    /* params + workspace on GPU */
    CudaParams p = cuda_params_create(HIDDEN);
    upload_bin_to_device(WEIGHTS_DIR "init_W1.bin", p.W1, (size_t)784*HIDDEN);
    upload_bin_to_device(WEIGHTS_DIR "init_b1.bin", p.b1, (size_t)HIDDEN);
    upload_bin_to_device(WEIGHTS_DIR "init_W2.bin", p.W2, (size_t)HIDDEN*10);
    upload_bin_to_device(WEIGHTS_DIR "init_b2.bin", p.b2, (size_t)10);

    CudaWorkSpace ws = cuda_ws_create(BATCH_SIZE, HIDDEN);
    upload_bin_to_device(LOGS_DIR "batch0_X.bin", ws.X, (size_t)BATCH_SIZE*784);
    upload_bin_to_device(LOGS_DIR "batch0_Y.bin", ws.Y, (size_t)BATCH_SIZE*10);

    cuda_forward (cub, &p, &ws);
    cuda_backward(cub, &p, &ws);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("=== Forward pass (fp32 GPU vs fp64 Python, tol=%.0e) ===\n", TOL);
    int ok = 1;
    ok &= check("Z1",  ws.Z1,  LOGS_DIR "batch0_Z1.bin",  (size_t)BATCH_SIZE*HIDDEN, TOL);
    ok &= check("A1",  ws.A1,  LOGS_DIR "batch0_A1.bin",  (size_t)BATCH_SIZE*HIDDEN, TOL);
    ok &= check("Z2",  ws.Z2,  LOGS_DIR "batch0_Z2.bin",  (size_t)BATCH_SIZE*10,     TOL);
    ok &= check("A2",  ws.A2,  LOGS_DIR "batch0_A2.bin",  (size_t)BATCH_SIZE*10,     TOL);

    printf("\n=== Backward pass ===\n");
    ok &= check("dW1", ws.dW1, LOGS_DIR "batch0_dW1.bin", (size_t)784*HIDDEN,        TOL);
    ok &= check("db1", ws.db1, LOGS_DIR "batch0_db1.bin", (size_t)HIDDEN,            TOL);
    ok &= check("dW2", ws.dW2, LOGS_DIR "batch0_dW2.bin", (size_t)HIDDEN*10,         TOL);
    ok &= check("db2", ws.db2, LOGS_DIR "batch0_db2.bin", (size_t)10,                TOL);

    printf("\n%s\n", ok ? "All checks PASSED." : "SOME CHECKS FAILED.");

    cuda_ws_free(&ws);
    cuda_params_free(&p);
    cublasDestroy(cub);
    return ok ? 0 : 1;
}
