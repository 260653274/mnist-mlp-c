#!/usr/bin/env bash
# benchmark.sh — compare naive C vs OpenBLAS vs Python across 50 epochs
# Usage: ./benchmark.sh

set -euo pipefail
cd "$(dirname "$0")"

LOG=logs/benchmark_results.txt
mkdir -p logs

echo "========================================" | tee "$LOG"
echo " MNIST MLP Benchmark — $(date)"          | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── helper: extract timing from last log line ────────────────────────
last_epoch_time() {
    # Reads train_log output and returns total time from the last epoch line
    grep "epoch  50" "$1" | awk '{print $NF}' | tr -d 's'
}

last_epoch_acc() {
    grep "epoch  50" "$1" | grep -oP 'test_acc=\K[0-9.]+' | tr -d '%'
}

# ── Python baseline ──────────────────────────────────────────────────
echo ">>> Python (numpy float64, 50 epochs)" | tee -a "$LOG"
PY_LOG=logs/bench_python.txt
conda run -n mnist-mlp python3 python/train.py 2>&1 | tee "$PY_LOG"
PY_TIME=$(grep "epoch  50" "$PY_LOG" | awk '{print $NF}' | tr -d 's')
PY_ACC=$(grep "epoch  50" "$PY_LOG" | grep -oP 'test_acc=\K[0-9.]+' | tr -d '%')
echo "" | tee -a "$LOG"

# ── Naive C ──────────────────────────────────────────────────────────
echo ">>> C Naive (triple-loop, 50 epochs)" | tee -a "$LOG"
NAIVE_LOG=logs/bench_naive.txt
./train_mnist 2>&1 | tee "$NAIVE_LOG"
NAIVE_TIME=$(grep "epoch  50" "$NAIVE_LOG" | awk '{print $NF}' | tr -d 's')
NAIVE_ACC=$(grep "epoch  50" "$NAIVE_LOG" | grep -oP 'test_acc=\K[0-9.]+' | tr -d '%')
echo "" | tee -a "$LOG"

# ── OpenBLAS C ───────────────────────────────────────────────────────
echo ">>> C + OpenBLAS (50 epochs)" | tee -a "$LOG"
BLAS_LOG=logs/bench_blas.txt
./train_mnist_blas 2>&1 | tee "$BLAS_LOG"
BLAS_TIME=$(grep "epoch  50" "$BLAS_LOG" | awk '{print $NF}' | tr -d 's')
BLAS_ACC=$(grep "epoch  50" "$BLAS_LOG" | grep -oP 'test_acc=\K[0-9.]+' | tr -d '%')
echo "" | tee -a "$LOG"

# ── Summary ──────────────────────────────────────────────────────────
NAIVE_VS_PY=$(echo "scale=1; $NAIVE_TIME / $PY_TIME" | bc)
BLAS_VS_PY=$(echo  "scale=2; $BLAS_TIME  / $PY_TIME"  | bc)
BLAS_VS_NAIVE=$(echo "scale=1; $NAIVE_TIME / $BLAS_TIME" | bc)

{
echo "========================================"
echo " Summary"
echo "========================================"
printf "%-20s  %8s  %10s  %10s\n" "Implementation" "Time(s)" "test_acc" "vs Python"
printf "%-20s  %8s  %10s  %10s\n" "--------------------" "-------" "--------" "---------"
printf "%-20s  %8s  %9s%%  %9sx\n" "Python (numpy)"   "$PY_TIME"    "$PY_ACC"    "1.00"
printf "%-20s  %8s  %9s%%  %9sx\n" "C Naive"          "$NAIVE_TIME" "$NAIVE_ACC" "${NAIVE_VS_PY}"
printf "%-20s  %8s  %9s%%  %9sx\n" "C + OpenBLAS"     "$BLAS_TIME"  "$BLAS_ACC"  "${BLAS_VS_PY}"
echo ""
echo "OpenBLAS speedup over Naive: ${BLAS_VS_NAIVE}x"
echo "========================================"
} | tee -a "$LOG"
