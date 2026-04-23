/* GPU-resident training of the 784-512-10 MLP.
 *
 * All tensors (dataset, weights, workspace) live on the GPU.  No H2D/D2H
 * copies happen inside the hot training loop — only the shuffle index
 * (1 int per training sample per epoch) goes up, and a single float
 * (epoch loss sum) plus a single int (correct count) come back.
 */

#include "nn_cuda.cuh"
#include "kernels.cuh"
#include "cuda_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

extern "C" {
#include "../src/data.h"
#include "../src/rng.h"
}

/* ── hyper-parameters ─────────────────────────────────────────────── */
#define HIDDEN      512
#define LR          0.09f
#define BATCH_SIZE  50
#define EPOCHS      50
#define SEED        42u
#define CE_EPS      1e-8f
#define EVAL_ROWS   10000   /* whole test set in one shot */

/* ── paths ────────────────────────────────────────────────────────── */
#define DATA_DIR    "data/"
#define WEIGHTS_DIR "weights/"
#define LOGS_DIR    "logs/"

/* ─────────────────────────── small helpers ───────────────────────── */

/* Convert an array of float64 to float32 in a fresh host buffer. */
static float *fp64_to_fp32(const double *src, size_t n) {
    float *dst = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
    return dst;
}

static void save_log(const char *path, double *losses, double *accs, int epochs) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write log %s\n", path); return; }
    fprintf(f, "epoch,loss,test_acc\n");
    for (int e = 0; e < epochs; e++)
        fprintf(f, "%d,%.8f,%.6f\n", e+1, losses[e], accs[e]);
    fclose(f);
}

/* Load a float64 .bin file and upload as float32 to a freshly-allocated
   device buffer.  Used for Python-exported init weights. */
static float *load_bin_fp32_device(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    double *h64 = (double *)malloc(n * sizeof(double));
    if (fread(h64, sizeof(double), n, f) != n) {
        fprintf(stderr, "short read %s\n", path); exit(1);
    }
    fclose(f);
    float *h32 = fp64_to_fp32(h64, n);
    free(h64);

    float *d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h32, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h32);
    return d;
}

/* Save a device float32 buffer back to host as float64 .bin, matching
   the weights/ .bin format used by the CPU targets. */
static void save_device_as_fp64(const char *path, const float *d, size_t n) {
    float *h32 = (float *)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h32, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    double *h64 = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++) h64[i] = (double)h32[i];
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite(h64, sizeof(double), n, f);
    fclose(f);
    free(h32); free(h64);
}

/* He init on host (float32) with our deterministic MT19937. */
static void he_init_host(float *W1, float *W2, int hidden, uint32_t seed) {
    RNG rng; rng_seed(&rng, seed);
    float std1 = (float)sqrt(2.0 / 784.0);
    for (int i = 0; i < 784 * hidden; i++)
        W1[i] = (float)rng_normal(&rng) * std1;
    float std2 = (float)sqrt(2.0 / hidden);
    for (int i = 0; i < hidden * 10; i++)
        W2[i] = (float)rng_normal(&rng) * std2;
}

/* ────────────────────────────── main ─────────────────────────────── */

int main(int argc, char *argv[]) {
    int load_init = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--load-init") == 0) load_init = 1;

    /* ── CUDA setup ───────────────────────────────────────────────── */
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  (sm_%d%d, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);

    cublasHandle_t cub;
    CUBLAS_CHECK(cublasCreate(&cub));

    /* ── load dataset on host ─────────────────────────────────────── */
    printf("Loading MNIST...\n");
    MNISTData train = mnist_load(DATA_DIR "train-images-idx3-ubyte",
                                 DATA_DIR "train-labels-idx1-ubyte");
    MNISTData test  = mnist_load(DATA_DIR "t10k-images-idx3-ubyte",
                                 DATA_DIR "t10k-labels-idx1-ubyte");
    printf("  train=%d  test=%d\n", train.n, test.n);

    /* ── upload dataset to GPU (one-shot, float32) ───────────────── */
    float *d_train_X, *d_train_Y, *d_test_X;
    int   *d_test_labels;
    {
        float *tmp = fp64_to_fp32(train.images, (size_t)train.n * 784);
        CUDA_CHECK(cudaMalloc(&d_train_X, (size_t)train.n * 784 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_train_X, tmp,
                              (size_t)train.n * 784 * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(tmp);

        tmp = fp64_to_fp32(train.labels_oh, (size_t)train.n * 10);
        CUDA_CHECK(cudaMalloc(&d_train_Y, (size_t)train.n * 10 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_train_Y, tmp,
                              (size_t)train.n * 10 * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(tmp);

        tmp = fp64_to_fp32(test.images, (size_t)test.n * 784);
        CUDA_CHECK(cudaMalloc(&d_test_X, (size_t)test.n * 784 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_test_X, tmp,
                              (size_t)test.n * 784 * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(tmp);

        CUDA_CHECK(cudaMalloc(&d_test_labels, (size_t)test.n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_test_labels, test.labels,
                              (size_t)test.n * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    printf("Dataset uploaded: train %.1f MB, test %.1f MB\n",
           train.n * 784 * 4 / 1e6, test.n * 784 * 4 / 1e6);

    /* ── allocate params + workspaces on GPU ─────────────────────── */
    CudaParams p = cuda_params_create(HIDDEN);

    if (load_init) {
        printf("Loading He init weights from Python export...\n");
        cudaFree(p.W1); p.W1 = load_bin_fp32_device(WEIGHTS_DIR "init_W1.bin", (size_t)784*HIDDEN);
        cudaFree(p.b1); p.b1 = load_bin_fp32_device(WEIGHTS_DIR "init_b1.bin", (size_t)HIDDEN);
        cudaFree(p.W2); p.W2 = load_bin_fp32_device(WEIGHTS_DIR "init_W2.bin", (size_t)HIDDEN*10);
        cudaFree(p.b2); p.b2 = load_bin_fp32_device(WEIGHTS_DIR "init_b2.bin", (size_t)10);
    } else {
        /* same deterministic MT19937 He init as CPU, but in fp32 */
        float *hW1 = (float*)malloc((size_t)784*HIDDEN*sizeof(float));
        float *hW2 = (float*)malloc((size_t)HIDDEN*10*sizeof(float));
        he_init_host(hW1, hW2, HIDDEN, SEED);
        CUDA_CHECK(cudaMemcpy(p.W1, hW1, (size_t)784*HIDDEN*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(p.W2, hW2, (size_t)HIDDEN*10*sizeof(float), cudaMemcpyHostToDevice));
        free(hW1); free(hW2);
        /* b1,b2 already zero from cuda_params_create */
        printf("He-initialised with seed %u (fp32)\n", SEED);
    }

    CudaWorkSpace ws = cuda_ws_create(BATCH_SIZE, HIDDEN);
    CudaEvalWS    es = cuda_evalws_create(EVAL_ROWS, HIDDEN);

    /* shuffle index buffers */
    int *h_idx = (int *)malloc((size_t)train.n * sizeof(int));
    int *d_idx;
    CUDA_CHECK(cudaMalloc(&d_idx, (size_t)train.n * sizeof(int)));

    /* per-epoch scalars */
    float *d_loss_sum; CUDA_CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    int   *d_correct;  CUDA_CHECK(cudaMalloc(&d_correct,  sizeof(int)));

    double *log_loss = (double *)malloc(EPOCHS * sizeof(double));
    double *log_acc  = (double *)malloc(EPOCHS * sizeof(double));

    RNG shuffle_rng; rng_seed(&shuffle_rng, SEED);

    /* ── training loop ────────────────────────────────────────────── */
    printf("\nTraining: epochs=%d  batch=%d  lr=%.4f  (FP32 on GPU)\n\n",
           EPOCHS, BATCH_SIZE, LR);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < train.n; i++) h_idx[i] = i;
        rng_shuffle(&shuffle_rng, h_idx, train.n);
        CUDA_CHECK(cudaMemcpy(d_idx, h_idx, (size_t)train.n*sizeof(int),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemsetAsync(d_loss_sum, 0, sizeof(float)));

        int n_batches = 0;
        for (int start = 0; start + BATCH_SIZE <= train.n; start += BATCH_SIZE) {
            /* gather X (bs,784) and Y (bs,10) from pre-uploaded dataset */
            launch_gather_rows(d_train_X, d_idx + start, ws.X, BATCH_SIZE, 784);
            launch_gather_rows(d_train_Y, d_idx + start, ws.Y, BATCH_SIZE, 10);

            cuda_forward(cub, &p, &ws);
            launch_ce_loss(ws.A2, ws.Y, d_loss_sum, BATCH_SIZE, 10, CE_EPS);
            cuda_backward(cub, &p, &ws);
            cuda_update(&p, &ws, LR);
            n_batches++;
        }

        /* ── evaluate test accuracy on GPU ────────────────────────── */
        CUDA_CHECK(cudaMemsetAsync(d_correct, 0, sizeof(int)));
        cuda_eval_forward(cub, &p, d_test_X, &es, test.n);
        launch_argmax_count(es.A2, d_test_labels, d_correct, test.n, 10, 0);

        float h_loss_sum; int h_correct;
        CUDA_CHECK(cudaMemcpy(&h_loss_sum, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_correct,  d_correct,  sizeof(int),   cudaMemcpyDeviceToHost));

        double avg_loss = (double)h_loss_sum / (double)(n_batches * BATCH_SIZE);
        double acc      = (double)h_correct / (double)test.n;
        log_loss[epoch] = avg_loss;
        log_acc[epoch]  = acc;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
        printf("epoch %3d/%d  loss=%.6f  test_acc=%.2f%%  t=%.2fs\n",
               epoch+1, EPOCHS, avg_loss, acc*100.0, elapsed);
    }

    /* ── save log and weights ─────────────────────────────────────── */
    save_log(LOGS_DIR "train_log_cuda.csv", log_loss, log_acc, EPOCHS);
    printf("\nLog saved to " LOGS_DIR "train_log_cuda.csv\n");

    save_device_as_fp64(WEIGHTS_DIR "cuda_W1.bin", p.W1, (size_t)784*HIDDEN);
    save_device_as_fp64(WEIGHTS_DIR "cuda_b1.bin", p.b1, (size_t)HIDDEN);
    save_device_as_fp64(WEIGHTS_DIR "cuda_W2.bin", p.W2, (size_t)HIDDEN*10);
    save_device_as_fp64(WEIGHTS_DIR "cuda_b2.bin", p.b2, (size_t)10);
    printf("Weights saved to " WEIGHTS_DIR "cuda_*.bin\n");

    /* ── cleanup ─────────────────────────────────────────────────── */
    free(h_idx); free(log_loss); free(log_acc);
    cudaFree(d_idx); cudaFree(d_loss_sum); cudaFree(d_correct);
    cudaFree(d_train_X); cudaFree(d_train_Y);
    cudaFree(d_test_X);  cudaFree(d_test_labels);
    cuda_ws_free(&ws);
    cuda_evalws_free(&es);
    cuda_params_free(&p);
    cublasDestroy(cub);
    mnist_free(&train); mnist_free(&test);
    return 0;
}
