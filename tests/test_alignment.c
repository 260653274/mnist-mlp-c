/*
 * Numerical alignment test: loads Python-exported initial weights and
 * batch-0 inputs, runs one forward+backward pass, then compares every
 * intermediate against Python's ground truth.
 *
 * Pass criterion: max absolute error < TOL for every tensor.
 *
 * Run AFTER python/npy_to_bin.py has converted all .npy files to .bin.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../src/matrix.h"
#include "../src/nn.h"
#include "../src/activation.h"

#define HIDDEN     512
#define BATCH_SIZE 50
#define TOL        1e-9   /* double precision; Python also uses float64 */

#define WEIGHTS_DIR "weights/"
#define LOGS_DIR    "logs/"

static double max_abs_err(const Matrix *A, const Matrix *B) {
    int n = A->rows * A->cols;
    double e = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(A->data[i] - B->data[i]);
        if (d > e) e = d;
    }
    return e;
}

static int check(const char *name, const Matrix *got, const Matrix *ref, double tol) {
    double e = max_abs_err(got, ref);
    int pass = e < tol;
    printf("  %-12s  max_err=%.2e  %s\n", name, e, pass ? "PASS" : "FAIL ***");
    return pass;
}

int main(void) {
    /* ── load Python initial weights ─────────────────────────── */
    Params p = params_create(HIDDEN);
    mat_free(&p.W1); p.W1 = mat_load_bin(WEIGHTS_DIR "init_W1.bin", 784,    HIDDEN);
    mat_free(&p.b1); p.b1 = mat_load_bin(WEIGHTS_DIR "init_b1.bin", 1,      HIDDEN);
    mat_free(&p.W2); p.W2 = mat_load_bin(WEIGHTS_DIR "init_W2.bin", HIDDEN, 10);
    mat_free(&p.b2); p.b2 = mat_load_bin(WEIGHTS_DIR "init_b2.bin", 1,      10);

    /* ── load batch-0 input ──────────────────────────────────── */
    WorkSpace ws = ws_create(BATCH_SIZE, HIDDEN);
    mat_free(&ws.X); ws.X = mat_load_bin(LOGS_DIR "batch0_X.bin", BATCH_SIZE, 784);
    mat_free(&ws.Y); ws.Y = mat_load_bin(LOGS_DIR "batch0_Y.bin", BATCH_SIZE, 10);

    /* ── forward pass ────────────────────────────────────────── */
    forward(&p, &ws);

    /* ── backward pass ───────────────────────────────────────── */
    backward(&p, &ws);

    /* ── load Python reference tensors ───────────────────────── */
    Matrix ref_Z1  = mat_load_bin(LOGS_DIR "batch0_Z1.bin",  BATCH_SIZE, HIDDEN);
    Matrix ref_A1  = mat_load_bin(LOGS_DIR "batch0_A1.bin",  BATCH_SIZE, HIDDEN);
    Matrix ref_Z2  = mat_load_bin(LOGS_DIR "batch0_Z2.bin",  BATCH_SIZE, 10);
    Matrix ref_A2  = mat_load_bin(LOGS_DIR "batch0_A2.bin",  BATCH_SIZE, 10);
    Matrix ref_dW1 = mat_load_bin(LOGS_DIR "batch0_dW1.bin", 784,        HIDDEN);
    Matrix ref_db1 = mat_load_bin(LOGS_DIR "batch0_db1.bin", 1,          HIDDEN);
    Matrix ref_dW2 = mat_load_bin(LOGS_DIR "batch0_dW2.bin", HIDDEN,     10);
    Matrix ref_db2 = mat_load_bin(LOGS_DIR "batch0_db2.bin", 1,          10);

    /* ── compare ─────────────────────────────────────────────── */
    printf("\n=== Forward pass ===\n");
    int ok = 1;
    ok &= check("Z1",  &ws.Z1, &ref_Z1,  TOL);
    ok &= check("A1",  &ws.A1, &ref_A1,  TOL);
    ok &= check("Z2",  &ws.Z2, &ref_Z2,  TOL);
    ok &= check("A2",  &ws.A2, &ref_A2,  TOL);

    printf("\n=== Backward pass ===\n");
    ok &= check("dW1", &ws.dW1, &ref_dW1, TOL);
    ok &= check("db1", &ws.db1, &ref_db1, TOL);
    ok &= check("dW2", &ws.dW2, &ref_dW2, TOL);
    ok &= check("db2", &ws.db2, &ref_db2, TOL);

    printf("\n%s\n", ok ? "All checks PASSED." : "SOME CHECKS FAILED.");

    /* ── cleanup ─────────────────────────────────────────────── */
    mat_free(&ref_Z1); mat_free(&ref_A1); mat_free(&ref_Z2); mat_free(&ref_A2);
    mat_free(&ref_dW1);mat_free(&ref_db1);mat_free(&ref_dW2);mat_free(&ref_db2);
    ws_free(&ws);
    params_free(&p);
    return ok ? 0 : 1;
}
