# gpu_test.py
import sys

print("Python version:", sys.version)

# --- Test CuPy ---
try:
    import cupy as cp
    print(" CuPy version:", cp.__version__)
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"   GPU detected: {dev['name'].decode()} (Compute Capability {dev['major']}.{dev['minor']})")

    # simple GPU computation
    x = cp.random.randn(2000, 2000, dtype=cp.float32)
    y = x @ x.T
    print("   CuPy matmul OK:", y.shape, "dtype:", y.dtype)

except Exception as e:
    print(" CuPy test failed:", e)


# --- Test NumPy ---
try:
    import numpy as np
    print(" NumPy version:", np.__version__)
    arr = np.arange(5)
    print("   NumPy array:", arr)
except Exception as e:
    print("NumPy test failed:", e)


# --- Test SciPy ---
try:
    import scipy
    from scipy import sparse
    print("SciPy version:", scipy.__version__)

    # sparse matrix CPU-side
    A = sparse.random(1000, 1000, density=0.01, format="csr")
    v = np.random.randn(1000)
    res = A @ v
    print("   SciPy sparse matvec OK:", res.shape)

    # optional: check GPU <-> CPU sparse copy
    from cupyx.scipy import sparse as cpsparse
    A_gpu = cpsparse.csr_matrix(A)
    v_gpu = cp.asarray(v)
    res_gpu = A_gpu @ v_gpu
    print(" cupyx.scipy.sparse matvec OK:", res_gpu.shape, "on GPU:", isinstance(res_gpu, cp.ndarray))

except Exception as e:
    print("SciPy test failed:", e)


# --- Test Optuna (optional) ---
try:
    import optuna
    print("Optuna version:", optuna.__version__)
except ImportError:
    print("ptuna not installed (only needed for kernel tuning).")
