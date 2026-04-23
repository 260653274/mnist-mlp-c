#include "rng.h"
#include <math.h>

/* Standard MT19937 reference implementation */
#define MT_N        624
#define MT_M        397
#define MATRIX_A    0x9908b0dfUL
#define UPPER_MASK  0x80000000UL
#define LOWER_MASK  0x7fffffffUL

void rng_seed(RNG *rng, uint32_t seed) {
    rng->mt[0] = seed;
    for (int i = 1; i < MT_N; i++)
        rng->mt[i] = (1812433253UL * (rng->mt[i-1] ^ (rng->mt[i-1] >> 30)) + i)
                     & 0xffffffffUL;
    rng->mti = MT_N;
}

uint32_t rng_uint32(RNG *rng) {
    static const uint32_t mag01[2] = {0x0UL, MATRIX_A};
    uint32_t y;

    if (rng->mti >= MT_N) {
        int kk;
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (rng->mt[kk] & UPPER_MASK) | (rng->mt[kk+1] & LOWER_MASK);
            rng->mt[kk] = rng->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 1];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (rng->mt[kk] & UPPER_MASK) | (rng->mt[kk+1] & LOWER_MASK);
            rng->mt[kk] = rng->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 1];
        }
        y = (rng->mt[MT_N-1] & UPPER_MASK) | (rng->mt[0] & LOWER_MASK);
        rng->mt[MT_N-1] = rng->mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 1];
        rng->mti = 0;
    }

    y = rng->mt[rng->mti++];
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

/* 53-bit precision uniform in [0,1) */
double rng_double(RNG *rng) {
    uint32_t a = rng_uint32(rng) >> 5;
    uint32_t b = rng_uint32(rng) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

/* Box-Muller: N(0,1).  Uses two uniforms per call; no caching (simple). */
double rng_normal(RNG *rng) {
    double u1, u2;
    do { u1 = rng_double(rng); } while (u1 == 0.0);
    u2 = rng_double(rng);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

void rng_shuffle(RNG *rng, int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        /* unbiased: only valid for i+1 <= 2^32, which holds for n=60000 */
        uint32_t r = rng_uint32(rng);
        int j = (int)(r % (uint32_t)(i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}
