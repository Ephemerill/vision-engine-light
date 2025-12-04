#!/bin/bash

# --- CRITICAL CONTAINER FIXES ---
# These MUST be set before Python starts to avoid Bus Errors

# 1. Threading Fixes (Stops "pthread_setaffinity_np" errors)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 2. Shared Memory Fixes (Stops "Bus error" / Shared Memory crashes)
# This forces the libraries to use standard RAM instead of the limited /dev/shm
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export ORT_TENSORRT_FP16_ENABLE=1 

echo "🚀 Starting Vision Engine with Container Fixes..."
python3 light.py