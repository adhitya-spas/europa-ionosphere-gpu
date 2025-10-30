# test_gpu_art_sanity.py
import sys, time

def main():
    print("Python:", sys.version.split()[0])

    import cupy as cp
    from cupyx.scipy import ndimage as cpx_ndimage
    from cupyx.scipy import sparse as cpx_sparse

    print(" CuPy:", cp.__version__)
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"   GPU: {dev['name'].decode()}  CC {dev['major']}.{dev['minor']}")

    # Optional CPU libs (for sparse round-trip test)
    try:
        import scipy
        from scipy import sparse as sp_sparse
        print(" SciPy:", scipy.__version__)
        have_scipy = True
    except Exception:
        have_scipy = False
        print(" SciPy: not installed (OK)")

    # ---------- Synthetic "truth" on GPU ----------
    n_lat, n_alt = 60, 80
    n_pix = n_lat * n_alt
    lat = cp.linspace(-10, 10, n_lat)
    alt = cp.linspace(50, 400, n_alt)
    LAT, ALT = cp.meshgrid(lat, alt)  # (n_alt, n_lat)

    def g2d(L, A, lat0, alt0, s_lat, s_alt, amp=1.0):
        return amp * cp.exp(-((L-lat0)**2/(2*s_lat**2) + (A-alt0)**2/(2*s_alt**2)))

    truth = (
        g2d(LAT, ALT, -1.0, 140.0, 1.0, 20.0, 1.0) +
        g2d(LAT, ALT,  2.5, 260.0, 1.2, 25.0, 0.8)
    ).astype(cp.float32)
    truth = cpx_ndimage.gaussian_filter(truth, sigma=(1.0, 1.0))
    x_true = truth.ravel(order="C")  # (n_pix,)

    # ---------- Random sparse geometry (GPU) ----------
    n_rays      = 8000
    nnz_per_ray = 30
    rng = cp.random.default_rng(123)

    col_centers = rng.integers(0, n_lat, size=n_rays)
    rows = cp.empty(n_rays * nnz_per_ray, dtype=cp.int32)
    cols = cp.empty_like(rows)
    data = cp.empty_like(rows, dtype=cp.float32)

    k = 0
    for i in range(n_rays):
        c0 = col_centers[i]
        lat_idx = cp.clip(c0 + rng.integers(-2, 3, size=nnz_per_ray), 0, n_lat-1)
        alt_idx = rng.integers(0, n_alt, size=nnz_per_ray)
        pix_idx = alt_idx * n_lat + lat_idx
        m = nnz_per_ray
        rows[k:k+m] = i
        cols[k:k+m] = pix_idx
        data[k:k+m] = (1.0 + rng.random(m, dtype=cp.float32)).astype(cp.float32)
        k += m

    D = cpx_sparse.csr_matrix((data, (rows, cols)), shape=(n_rays, n_pix))
    print(f"Built GPU CSR geometry: {D.shape}, nnz={int(D.nnz)} (~{nnz_per_ray} per row)")

    if have_scipy:
        try:
            D_cpu = D.get()
            assert sp_sparse.isspmatrix_csr(D_cpu)
            print(" Sparse CPU<->GPU copy OK")
        except Exception as e:
            print(" Sparse copy warning:", e)

    # ---------- Simulate measurements y = D x_true + noise ----------
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    y = D @ x_true
    sigma = 0.02 * cp.linalg.norm(y) / cp.sqrt(y.size)
    y = (y + sigma * rng.standard_normal(y.shape, dtype=cp.float32)).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()
    print(f" Simulated measurements: {y.shape} in {time.time()-t0:.3f}s")

    # ---------- SIRT with row/column normalization ----------
    # SIRT update: x_{k+1} = x_k + λ * C^{-1} D^T ( R^{-1} (y - D x_k) )
    # R = diag(row_sums), C = diag(col_sums), with small eps for stability.
    eps = cp.float32(1e-8)
    row_sums = cp.asarray(D.sum(axis=1)).ravel().astype(cp.float32)
    col_sums = cp.asarray(D.sum(axis=0)).ravel().astype(cp.float32)  # length = n_pix

    Rinv = 1.0 / cp.maximum(row_sums, eps)      # (n_rays,)
    Cinv = 1.0 / cp.maximum(col_sums, eps)      # (n_pix,)

    DT = D.T.tocsr()
    x  = cp.zeros(n_pix, dtype=cp.float32)
    relax = cp.float32(0.2)   # gentler step size
    n_iters = 60

    cp.cuda.Stream.null.synchronize()
    t1 = time.time()
    for k in range(n_iters):
        r = y - D @ x                         # (n_rays,)
        z = r * Rinv                          # normalize by rows
        dx = DT @ z                           # backproject
        x  = x + relax * (dx * Cinv)          # normalize by columns and step
        # Optional: prevent blow-ups from negative values if your physics demands >=0
        x = cp.maximum(x, 0.0)

        if not cp.isfinite(x).all():
            print(f"non-finite detected at iter {k}")
            break
    cp.cuda.Stream.null.synchronize()
    print(f" SIRT finished in {time.time()-t1:.3f}s, iters={k+1}")

    # ---------- Metrics ----------
    x_rec = x.reshape((n_alt, n_lat))

    def rmse(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(cp.sqrt(cp.mean((a-b)**2)).get())

    def ncc(a, b):
        a0 = a - a.mean(); b0 = b - b.mean()
        denom = cp.linalg.norm(a0) * cp.linalg.norm(b0)
        num = (a0*b0).sum()
        denom_f = float(denom.get())
        return float((num/denom).get()) if denom_f > 0 else 0.0

    print(f" RMSE: {rmse(x_rec, truth):.3e}   NCC: {ncc(x_rec, truth):.3f}")

    preview = cp.asnumpy(x_rec[:3, :3])
    print(" GPU→CPU preview:\n", preview)
    print("\nGPU sanity checks completed successfully.")

if __name__ == "__main__":
    main()
