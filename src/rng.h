#pragma once
#include <stdint.h>

typedef struct {
    uint32_t mt[624];
    int mti;
} RNG;

void     rng_seed(RNG *rng, uint32_t seed);
uint32_t rng_uint32(RNG *rng);
double   rng_double(RNG *rng);   /* uniform [0, 1) */
double   rng_normal(RNG *rng);   /* standard normal N(0,1), Box-Muller */

/* Fisher-Yates shuffle on integer array of length n */
void rng_shuffle(RNG *rng, int *arr, int n);
