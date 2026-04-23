#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matrix.h"
#include "nn.h"
#include "data.h"
#include "rng.h"
#include "activation.h"

/* ── hyper-parameters ─────────────────────────────────────── */
#define HIDDEN      512
#define LR          0.09
#define BATCH_SIZE  50
#define EPOCHS      50
#define SEED        42u
#define CHUNK_SIZE  10000  /* full test set in one GEMM — maximises BLAS/OMP benefit */

/* ── paths ─────────────────────────────────────────────────── */
#define DATA_DIR    "data/"
#define WEIGHTS_DIR "weights/"
#define LOGS_DIR    "logs/"

static void save_log(const char *path, double *losses, double *accs, int epochs) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write log %s\n", path); return; }
    fprintf(f, "epoch,loss,test_acc\n");
    for (int e = 0; e < epochs; e++)
        fprintf(f, "%d,%.8f,%.6f\n", e+1, losses[e], accs[e]);
    fclose(f);
}

int main(int argc, char *argv[]) {
    /* optional flag: --load-init  loads He weights from Python export */
    int load_init = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--load-init") == 0) load_init = 1;

    /* ── load data ─────────────────────────────────────────── */
    printf("Loading MNIST...\n");
    MNISTData train = mnist_load(DATA_DIR "train-images-idx3-ubyte",
                                 DATA_DIR "train-labels-idx1-ubyte");
    MNISTData test  = mnist_load(DATA_DIR "t10k-images-idx3-ubyte",
                                 DATA_DIR "t10k-labels-idx1-ubyte");
    printf("  train=%d  test=%d\n", train.n, test.n);

    /* ── init params ────────────────────────────────────────── */
    Params p = params_create(HIDDEN);
    if (load_init) {
        printf("Loading initial weights from Python export...\n");
        mat_free(&p.W1); p.W1 = mat_load_bin(WEIGHTS_DIR "init_W1.bin", 784,   HIDDEN);
        mat_free(&p.b1); p.b1 = mat_load_bin(WEIGHTS_DIR "init_b1.bin", 1,     HIDDEN);
        mat_free(&p.W2); p.W2 = mat_load_bin(WEIGHTS_DIR "init_W2.bin", HIDDEN, 10);
        mat_free(&p.b2); p.b2 = mat_load_bin(WEIGHTS_DIR "init_b2.bin", 1,     10);
    } else {
        RNG rng;
        rng_seed(&rng, SEED);
        params_he_init(&p, &rng);
        printf("He-initialised with seed %u\n", SEED);
    }

    /* ── pre-allocate workspace ─────────────────────────────── */
    WorkSpace ws = ws_create(BATCH_SIZE, HIDDEN);
    int *idx = (int *)malloc(train.n * sizeof(int));

    double *log_loss = (double *)malloc(EPOCHS * sizeof(double));
    double *log_acc  = (double *)malloc(EPOCHS * sizeof(double));

    RNG shuffle_rng;
    rng_seed(&shuffle_rng, SEED);

    /* ── training loop ──────────────────────────────────────── */
    printf("\nTraining: epochs=%d  batch=%d  lr=%.4f\n\n",
           EPOCHS, BATCH_SIZE, LR);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        /* shuffle */
        for (int i = 0; i < train.n; i++) idx[i] = i;
        rng_shuffle(&shuffle_rng, idx, train.n);

        double epoch_loss = 0.0;
        int n_batches = 0;

        for (int start = 0; start + BATCH_SIZE <= train.n; start += BATCH_SIZE) {
            /* gather batch into ws.X and ws.Y */
            for (int i = 0; i < BATCH_SIZE; i++) {
                int si = idx[start + i];
                memcpy(ws.X.data + i*784,
                       train.images    + (size_t)si * 784,
                       784 * sizeof(double));
                memcpy(ws.Y.data + i*10,
                       train.labels_oh + (size_t)si * 10,
                       10 * sizeof(double));
            }

            forward(&p, &ws);
            epoch_loss += cross_entropy_loss(&ws.A2, &ws.Y);
            backward(&p, &ws);
            params_update(&p, &ws, LR);
            n_batches++;
        }

        double avg_loss = epoch_loss / n_batches;
        double acc      = test_accuracy(&p, test.images, test.labels,
                                        test.n, CHUNK_SIZE, HIDDEN);
        log_loss[epoch] = avg_loss;
        log_acc[epoch]  = acc;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
        printf("epoch %3d/%d  loss=%.6f  test_acc=%.2f%%  t=%.1fs\n",
               epoch+1, EPOCHS, avg_loss, acc*100.0, elapsed);
    }

    /* ── save log ───────────────────────────────────────────── */
    save_log(LOGS_DIR "train_log_c.csv", log_loss, log_acc, EPOCHS);
    printf("\nLog saved to " LOGS_DIR "train_log_c.csv\n");

    /* ── save trained weights ───────────────────────────────── */
    mat_save_bin(WEIGHTS_DIR "c_W1.bin", &p.W1);
    mat_save_bin(WEIGHTS_DIR "c_b1.bin", &p.b1);
    mat_save_bin(WEIGHTS_DIR "c_W2.bin", &p.W2);
    mat_save_bin(WEIGHTS_DIR "c_b2.bin", &p.b2);
    printf("Weights saved to " WEIGHTS_DIR "\n");

    /* ── cleanup ────────────────────────────────────────────── */
    free(idx); free(log_loss); free(log_acc);
    ws_free(&ws);
    params_free(&p);
    mnist_free(&train);
    mnist_free(&test);
    return 0;
}
