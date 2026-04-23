CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -march=native -Isrc
LDFLAGS = -lm

# ── OpenBLAS via conda env ──────────────────────────────────────────
CONDA_ENV = $(HOME)/anaconda3/envs/mnist-mlp
BLAS_INC  = -I$(CONDA_ENV)/include
BLAS_LIB  = -L$(CONDA_ENV)/lib -lopenblas -Wl,-rpath,$(CONDA_ENV)/lib

OMP_FLAGS = -fopenmp

# ── CUDA ────────────────────────────────────────────────────────────
NVCC       = /usr/local/cuda-12.8/bin/nvcc
CUDA_ARCH  = -arch=native
NVCC_FLAGS = -O3 $(CUDA_ARCH) -std=c++14 -Icuda -Isrc \
             -Xcompiler -Wall,-Wextra,-O3
CUDA_LIBS  = -lcublas -lcudart
CUDA_KSRC  = cuda/kernels.cu cuda/nn_cuda.cu
CUDA_CSRC  = src/data.c src/rng.c

# ── source groups ───────────────────────────────────────────────────
# Shared sources — matrix ops (without GEMM), activations, nn, data, rng
COMMON_SRCS = src/matrix.c src/activation.c src/nn.c src/data.c src/rng.c
# Naive GEMM (triple-loop, OMP-aware via #pragma)
NAIVE_GEMM  = src/matrix_gemm.c
# BLAS GEMM (cblas_dgemm)
BLAS_GEMM   = blas/matrix_blas.c

# Object file sets per target
OBJS       = $(COMMON_SRCS:.c=.o)        $(NAIVE_GEMM:.c=.o)
OBJS_OMP   = $(COMMON_SRCS:.c=.omp.o)    $(NAIVE_GEMM:.c=.omp.o)
OBJS_BLAS  = $(COMMON_SRCS:.c=.o)        $(BLAS_GEMM:.c=.blas.o)

.PHONY: all clean cuda

all: train_mnist train_mnist_blas train_mnist_omp \
     test_alignment test_train_alignment \
     bench_gemm bench_gemm_blas bench_gemm_omp

cuda: train_mnist_cuda test_alignment_cuda

# ── naive C (triple-loop) ───────────────────────────────────────────
train_mnist: $(OBJS) src/train.c
	$(CC) $(CFLAGS) -o $@ src/train.c $(OBJS) $(LDFLAGS)

test_alignment: $(OBJS) tests/test_alignment.c
	$(CC) $(CFLAGS) -o $@ tests/test_alignment.c $(OBJS) $(LDFLAGS)

test_train_alignment: $(OBJS) tests/test_train_alignment.c
	$(CC) $(CFLAGS) -o $@ tests/test_train_alignment.c $(OBJS) $(LDFLAGS)

bench_gemm: $(OBJS) tests/bench_gemm.c
	$(CC) $(CFLAGS) -o $@ tests/bench_gemm.c $(OBJS) $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ── OpenMP (naive GEMM + #pragma omp parallel for) ──────────────────
train_mnist_omp: $(OBJS_OMP) src/train.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ src/train.c $(OBJS_OMP) $(LDFLAGS)

bench_gemm_omp: $(OBJS_OMP) tests/bench_gemm.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ tests/bench_gemm.c $(OBJS_OMP) $(LDFLAGS)

src/%.omp.o: src/%.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -c -o $@ $<

# ── OpenBLAS (blas/matrix_blas.c replaces naive GEMM) ───────────────
train_mnist_blas: $(OBJS_BLAS) src/train.c
	$(CC) $(CFLAGS) $(BLAS_INC) -o $@ src/train.c $(OBJS_BLAS) $(LDFLAGS) $(BLAS_LIB)

bench_gemm_blas: $(OBJS_BLAS) tests/bench_gemm.c
	$(CC) $(CFLAGS) $(BLAS_INC) -o $@ tests/bench_gemm.c $(OBJS_BLAS) $(LDFLAGS) $(BLAS_LIB)

blas/%.blas.o: blas/%.c
	$(CC) $(CFLAGS) $(BLAS_INC) -c -o $@ $<

# ── CUDA (GPU-resident, FP32, cuBLAS + custom kernels) ──────────────
train_mnist_cuda: cuda/train_cuda.cu $(CUDA_KSRC) $(CUDA_CSRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ \
	    cuda/train_cuda.cu $(CUDA_KSRC) $(CUDA_CSRC) \
	    $(CUDA_LIBS)

test_alignment_cuda: cuda/test_alignment_cuda.cu $(CUDA_KSRC) $(CUDA_CSRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ \
	    cuda/test_alignment_cuda.cu $(CUDA_KSRC) $(CUDA_CSRC) \
	    $(CUDA_LIBS)

clean:
	rm -f $(OBJS) $(OBJS_OMP) $(OBJS_BLAS) \
	      train_mnist train_mnist_blas train_mnist_omp \
	      test_alignment test_train_alignment \
	      bench_gemm bench_gemm_blas bench_gemm_omp \
	      train_mnist_cuda test_alignment_cuda
