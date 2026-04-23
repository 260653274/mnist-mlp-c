/*
 * 5-epoch training alignment test.
 * Loads Python's initial weights + shuffle indices, trains 5 epochs,
 * compares loss against Python's CSV log.
 * Pass criterion: relative loss error < 1% for all 5 epochs.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/matrix.h"
#include "../src/nn.h"
#include "../src/data.h"
#include "../src/activation.h"

#define HIDDEN     512
#define BATCH_SIZE 50
#define LR         0.09
#define N_EPOCHS   5
#define REL_TOL    0.01   /* 1% */

#define DATA_DIR    "data/"
#define WEIGHTS_DIR "weights/"
#define LOGS_DIR    "logs/"

/* Python reference losses (epoch 1-5 from logs/train_log.csv) */
static const double PY_LOSS[N_EPOCHS] = {
    0.331284, 0.170383, 0.123979, 0.098187, 0.079781
};

static int32_t *load_shuffle(const char *path, int n) {
    int32_t *idx = (int32_t *)malloc(n * sizeof(int32_t));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(idx, sizeof(int32_t), n, f) != (size_t)n)
        { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return idx;
}

int main(void) {
    /* ── load data ────────────────────────────────────────── */
    MNISTData train = mnist_load(DATA_DIR "train-images-idx3-ubyte",
                                 DATA_DIR "train-labels-idx1-ubyte");

    /* ── load Python initial weights ─────────────────────── */
    Params p = params_create(HIDDEN);
    mat_free(&p.W1); p.W1 = mat_load_bin(WEIGHTS_DIR "init_W1.bin", 784,    HIDDEN);
    mat_free(&p.b1); p.b1 = mat_load_bin(WEIGHTS_DIR "init_b1.bin", 1,      HIDDEN);
    mat_free(&p.W2); p.W2 = mat_load_bin(WEIGHTS_DIR "init_W2.bin", HIDDEN, 10);
    mat_free(&p.b2); p.b2 = mat_load_bin(WEIGHTS_DIR "init_b2.bin", 1,      10);

    /* ── allocate workspace ──────────────────────────────── */
    WorkSpace ws = ws_create(BATCH_SIZE, HIDDEN);

    printf("\n=== 5-epoch training alignment (Python init weights + shuffles) ===\n");
    printf("%-6s  %-12s  %-12s  %-10s  %s\n",
           "epoch", "py_loss", "c_loss", "rel_err%", "status");
    printf("%-6s  %-12s  %-12s  %-10s  %s\n",
           "-----", "--------", "------", "--------", "------");

    int all_pass = 1;

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        /* load this epoch's shuffle from Python */
        char path[256];
        snprintf(path, sizeof(path), LOGS_DIR "shuffle_ep%d.bin", epoch+1);
        int32_t *idx = load_shuffle(path, train.n);

        double epoch_loss = 0.0;
        int    n_batches  = 0;

        for (int start = 0; start + BATCH_SIZE <= train.n; start += BATCH_SIZE) {
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
        free(idx);

        double avg_loss = epoch_loss / n_batches;
        double py_loss  = PY_LOSS[epoch];
        double rel_err  = fabs(avg_loss - py_loss) / py_loss * 100.0;
        int    pass     = rel_err < REL_TOL * 100.0;
        if (!pass) all_pass = 0;

        printf("%-6d  %-12.6f  %-12.6f  %-9.4f%%  %s\n",
               epoch+1, py_loss, avg_loss, rel_err,
               pass ? "PASS" : "FAIL ***");
    }

    printf("\n%s\n", all_pass ? "All 5-epoch checks PASSED." : "SOME CHECKS FAILED.");

    ws_free(&ws);
    params_free(&p);
    mnist_free(&train);
    return all_pass ? 0 : 1;
}
