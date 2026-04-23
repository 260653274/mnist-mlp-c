/*
 * Micro-benchmark: compare naive vs OpenBLAS mat_mul at different M sizes.
 * Keeps K=784, N=512 fixed (our W1 dimensions); varies M (batch size).
 * This shows the crossover point where BLAS becomes faster.
 *
 * Build:  make bench_gemm  (naive) or  make bench_gemm_blas  (BLAS)
 * Usage:  ./bench_gemm
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../src/matrix.h"
#include "../src/rng.h"

#define K 784
#define N 512
#define REPS 200   /* repetitions per M size */

static double elapsed_ms(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec)*1000.0 + (b.tv_nsec - a.tv_nsec)*1e-6;
}

int main(void) {
    RNG rng; rng_seed(&rng, 123);

    int M_vals[] = {1, 10, 50, 100, 200, 500, 1000, 2000, 5000};
    int n_vals   = sizeof(M_vals)/sizeof(M_vals[0]);

    printf("%-8s  %-10s  %-12s  %-10s\n",
           "M", "GFLOP/s", "ms/call", "total_ms");
    printf("%-8s  %-10s  %-12s  %-10s\n",
           "----", "-------", "-------", "--------");

    for (int vi = 0; vi < n_vals; vi++) {
        int M = M_vals[vi];
        Matrix A = mat_create(M, K);
        Matrix B = mat_create(K, N);
        Matrix C = mat_create(M, N);

        /* fill with random data */
        for (int i = 0; i < M*K; i++) A.data[i] = rng_normal(&rng) * 0.1;
        for (int i = 0; i < K*N; i++) B.data[i] = rng_normal(&rng) * 0.1;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < REPS; r++)
            mat_mul(&A, &B, &C);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double total_ms = elapsed_ms(t0, t1);
        double per_call = total_ms / REPS;
        double gflops   = (2.0 * M * K * N * REPS) / (total_ms * 1e6);  /* 2ops per MAC */

        printf("M=%-6d  %9.2f  %10.4f ms  %8.1f ms\n",
               M, gflops, per_call, total_ms);

        mat_free(&A); mat_free(&B); mat_free(&C);
    }
    return 0;
}
