#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t _err = (call);                                            \
    if (_err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(_err));            \
        exit(1);                                                          \
    }                                                                     \
} while (0)

#define CUBLAS_CHECK(call) do {                                           \
    cublasStatus_t _st = (call);                                          \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                       \
                __FILE__, __LINE__, (int)_st);                            \
        exit(1);                                                          \
    }                                                                     \
} while (0)
