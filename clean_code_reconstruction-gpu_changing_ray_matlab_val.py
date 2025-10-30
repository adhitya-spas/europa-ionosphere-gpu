"""
reconstruction_main_refactored.py

A tidy, step-by-step main script that mirrors your documented pipeline:
  Step 0) Load data
  Step 1) Build ionosphere
  Step 2) Generate rays (VHF nadir, HF oblique)
  Step 3) Forward model (STEC → dual-freq Δt → VTEC)
  Step 4) Build geometry D (path-length weighted)
  Step 5) Reconstruct (ART / Weighted ART / δ(Δt) branch)
  Step 6) Evaluate & plot

NOTE: This uses YOUR existing functions/types/conventions. No behavior changes.
"""

# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy
import scipy.sparse as sp 
import os
from datetime import datetime

from modify_df import load_mission_df
from ionosphere_design import (
    build_ionosphere, add_gaussian_enhancement
)
from ray_trace_passive import (
    trace_passive_nadir, trace_passive_oblique,
    compute_STEC_along_path, dualfreq_to_VTEC, reconstruct_art, reconstruct_art_sparse
)
from improved_geometry import (
    build_geometry_matrix_weighted_sparse,
    build_geometry_matrix_weighted,
    calculate_measurement_weights,
    weighted_reconstruction_art,
    weighted_reconstruction_art_sparse
)

# ---------------------- Config -----------------------
# VHF & HF radio parameters (same conventions as your current main)
F_C_VHF = 60e6        # VHF center frequency [Hz]
BW_VHF  = 10e6        # VHF bandwidth [Hz]
F_C_HF  = 9e6         # HF center frequency [Hz]
BW_HF   = 1e6         # HF bandwidth [Hz]

# Rays / geometry
NPTS_VHF = 500        # samples along each VHF ray
NPTS_HF  = 500        # samples along each HF ray (match VHF to keep ds small)
SAT_LATS_VHF = np.linspace(-10.0, 10.0, 60)  #np.linspace(-10.0, -5.0, 10)  # ground-track latitudes for rays
SAT_LATS_HF = np.linspace(-8, 8, 16)
RAY_TRACE_MODE = "left_fixed"  # "left_fixed" or "sat_right" or "mirror"

# HF true incidence angles (from vertical, 0° = nadir)
# THETA_LIST_HF = [30, 40, 50, 60, 70, 80, 85]
THETA_LIST_HF = np.linspace(5, 16, 20) # [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10]

# Integration times (seconds), per your current logic
INTEG_TIMES_VHF = 0.12
# INTEG_TIMES_HF_PER_ANGLE = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
INTEG_TIMES_HF_PER_ANGLE = np.linspace(0.05, 0.15, 100)  # 20 values from 0.05s to 0.15s

# ART settings
N_ITERS = 20
RELAX   = 0.1

# Some constants
C = 3e8  # speed of light [m/s]
K_IONO = 40.3  # ionospheric delay constant in SI: Δt = K_IONO * TEC / (c f^2)

# Create one timestamped folder per process start
IMG_TS = datetime.now().strftime("%m_%d_%H_%M_%S")
IMG_DIR = os.path.join("img", IMG_TS)
os.makedirs(IMG_DIR, exist_ok=True)

# Top-level smoothing toggle (0.0 = off)
SMOOTH_SIGMA = 1.2

TEST_CHECKS = False  # run unit test checks before main
# ---------------------- Step -1: GPU SUBSTITUTES ----------------------
USE_GPU = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import scipy.sparse as sp
except Exception:
    USE_GPU = False
    cp = None
    cpx_sparse = None

def build_passive_plot_inputs_GPU(
    iono, lats, alts_m,
    D_hf,                 # geometry (n_hf_rays x n_pix); dense OK, sparse preferred
    integ_hf,             # (n_hf_rays,) integration times [s]
    theta_per_ray,        # (n_hf_rays,) true incidence angles [deg]
    vtec_hf,              # (n_hf_rays,) VTEC estimates from your helper
    delta_t_hf,           # (n_hf_rays,) noisy HF dual-freq delay [s]
    delta_t_vhf,          # (n_vhf_rays,) noisy VHF dual-freq delay [s]
    sat_lats_vhf=np.linspace(-6.0, 6.0, 60),
    hf_origin_lat=None,
    F_C_HF=9e6, BW_HF=1e6, F_C_VHF=60e6,
):
    if not USE_GPU:
        raise RuntimeError("Set USE_GPU=True and ensure CuPy is available.")

    # ---- Push scalars/arrays to GPU (float32 is enough here) ----
    iono_gpu   = cp.asarray(iono, dtype=cp.float32)            # (n_alt, n_lat)
    x_gpu      = iono_gpu.ravel(order="C")                     # (n_pix,)
    integ_gpu  = cp.asarray(integ_hf, dtype=cp.float32)
    theta_gpu  = cp.asarray(theta_per_ray, dtype=cp.float32)
    vtec_hf_gp = cp.asarray(vtec_hf, dtype=cp.float32)
    dt_hf_gpu  = cp.asarray(delta_t_hf, dtype=cp.float32)
    dt_vhf_gpu = cp.asarray(delta_t_vhf, dtype=cp.float32)

    # ---- Geometry to GPU (sparse preferred) ----
    # If D_hf is dense (np.ndarray), convert to GPU sparse CSR to save memory
    # You can also build CSR on CPU (scipy.sparse.csr_matrix) then move to GPU.
    if hasattr(D_hf, "tocsr"):  # already SciPy sparse
        D_hf_gpu = cpx_sparse.csr_matrix(D_hf)
    else:
        # assume dense np.ndarray
        D_hf_gpu = cpx_sparse.csr_matrix(cp.asarray(D_hf))

    # ---- 2) True slant TEC per HF ray: tec_true = D_hf @ x ----
    tec_true_gpu = D_hf_gpu @ x_gpu     # (n_hf_rays,)

    # ---- 3A) Passive TEC estimator absolute error (m^-2) ----
    stec_est_gpu = vtec_hf_gp / cp.cos(cp.deg2rad(theta_gpu))
    delta_tec_abs_gpu = cp.abs(stec_est_gpu - tec_true_gpu)

    # ---- 3B) δ(Δt) residual → TEC → VHF range error ----
    c = cp.asarray(3e8, dtype=cp.float32)
    f1_h = cp.asarray(F_C_HF - 0.5*BW_HF, dtype=cp.float32)
    f2_h = cp.asarray(F_C_HF + 0.5*BW_HF, dtype=cp.float32)
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    # Match a single VHF Δt reference to all HF rays (same convention as your main)
    if hf_origin_lat is None:
        hf_origin_lat = float(sat_lats_vhf[0])
    vhf_idx = int(np.argmin(np.abs(np.asarray(sat_lats_vhf) - hf_origin_lat)))
    dt_vhf_match_gpu = cp.full_like(dt_hf_gpu, dt_vhf_gpu[vhf_idx])
    delta_dt_gpu = -(dt_vhf_match_gpu - dt_hf_gpu)

    tec_residual_gpu = (c * delta_dt_gpu / K_IONO) * freq_factor_h    # m^-2
    range_err_gpu    = (K_IONO / (cp.asarray(F_C_VHF, dtype=cp.float32)**2)) * cp.abs(tec_residual_gpu)

    # ---- Return NumPy for plotting, keep GPU speed for compute ----
    t_meas      = cp.asnumpy(integ_gpu)
    tec_for_bin = cp.asnumpy(tec_true_gpu)
    dtec_abs    = cp.asnumpy(delta_tec_abs_gpu)
    rng_err     = cp.asnumpy(range_err_gpu)
    return t_meas, tec_for_bin, dtec_abs, rng_err

def _gpu_binned_mean_2d(x_cpu, y_cpu, z_cpu, xb_cpu, yb_cpu):
    """
    Return (H_mean, H_count). Works even if inputs are empty.
    x,y define the grid; z are values to average per 2D bin.
    """

    x_cpu = np.asarray(x_cpu); y_cpu = np.asarray(y_cpu); z_cpu = np.asarray(z_cpu)
    xb_cpu = np.asarray(xb_cpu); yb_cpu = np.asarray(yb_cpu)

    # Sanity on bin edges
    if xb_cpu.size < 2 or yb_cpu.size < 2:
        raise ValueError("Bin edges must have length >= 2 in each dimension.")

    # Drop NaNs
    mask = np.isfinite(x_cpu) & np.isfinite(y_cpu) & np.isfinite(z_cpu)
    x_cpu, y_cpu, z_cpu = x_cpu[mask], y_cpu[mask], z_cpu[mask]

    # If nothing left, return zeros
    out_shape = (yb_cpu.size - 1, xb_cpu.size - 1)
    if x_cpu.size == 0:
        H = np.zeros(out_shape, dtype=float)
        C = np.zeros(out_shape, dtype=float)
        return H, C

    # GPU path if available
    if cp is not None:
        x = cp.asarray(x_cpu); y = cp.asarray(y_cpu); z = cp.asarray(z_cpu)
        xb = cp.asarray(xb_cpu); yb = cp.asarray(yb_cpu)

        # Compute weighted sum and counts with histogram2d
        H, _, _ = cp.histogram2d(y, x, bins=[yb, xb], weights=z)   # sum of z per bin
        C, _, _ = cp.histogram2d(y, x, bins=[yb, xb])              # count per bin

        # Avoid divide-by-zero
        C_nz = cp.where(C == 0, 1, C)
        M = H / C_nz

        return cp.asnumpy(M), cp.asnumpy(C)
    else:
        # CPU fallback (NumPy)
        H, _, _ = np.histogram2d(y_cpu, x_cpu, bins=[yb_cpu, xb_cpu], weights=z_cpu)
        C, _, _ = np.histogram2d(y_cpu, x_cpu, bins=[yb_cpu, xb_cpu])
        C_nz = np.where(C == 0, 1, C)
        M = H / C_nz
        return M, C

# ---------------------- Step 0: Load data ----------------------
def step0_load_dataframe(path="new_mission_df.pkl", mission_name="E6a Exit"):
    df = load_mission_df(path)
    row = df[df["Mission"] == mission_name].iloc[0]
    alt_km_1d, Ne_1d = row["Altitude"], row["Ne"]
    return alt_km_1d, Ne_1d

# ---------------------- Step 1: Build ionosphere ----------------
def step1_build_iono(alt_km_1d, Ne_1d,
                     lat_extent=(-10, 10), lat_res=200,
                     add_gaussian=True):
    # Build base map (extrude 1D profile to 2D)
    lats, alt_km, iono = build_ionosphere(
        pd.DataFrame({"altitude": alt_km_1d, "ne": Ne_1d}),
        lat_extent=lat_extent, lat_res=lat_res
    )

    # Optional Gaussian enhancement (same call pattern you already use)
    if add_gaussian:
        iono = add_gaussian_enhancement(
            ionosphere_map=iono,
            latitudes=lats,
            altitude=alt_km,
            lat_center=0.0,
            alt_center=150.0,
            lat_width=1.0,
            alt_width=20.0,
            amplitude=5e11
        )

    alts_m = alt_km * 1e3
    return lats, alt_km, alts_m, iono

# ---------------------- Step 2: Generate rays -------------------
def step2_generate_rays(lats, alts_m):
    # 2A) VHF nadir rays
    rays_vhf = trace_passive_nadir(SAT_LATS_VHF, alts_m, npts=NPTS_VHF)
    integ_vhf = np.full(len(rays_vhf), INTEG_TIMES_VHF)

    # 2B) HF oblique rays (true incidence angles from vertical)
    rays_hf_all = []
    theta_per_ray = []
    integ_hf_all = []

    T_PER_ANGLE = np.full(len(THETA_LIST_HF), 0.12, dtype=float)  # e.g. 0.12s
    print(T_PER_ANGLE)
    print(THETA_LIST_HF)


    for idx, theta in enumerate(THETA_LIST_HF):
        rays_hf = trace_passive_oblique(
            sat_lats=SAT_LATS_HF,
            h0_m=alts_m.max(),
            hsat_m=alts_m.max(),
            theta_i_deg=theta,
            npts=NPTS_HF,
            R_E_m=1.5608e6,
            mode=RAY_TRACE_MODE,
            lat_left_deg=0.0          # << choose your left start latitude
        )
        # rays_hf = trace_passive_oblique(
        #     sat_lats=SAT_LATS_HF,
        #     h0=alts_m.max(),
        #     hsat=alts_m.max(),
        #     theta_i_deg=theta,       # true incidence from vertical
        #     npts=NPTS_HF
        # )
        rays_hf_all.extend(rays_hf)
        theta_per_ray.extend([theta] * len(rays_hf))
        integ_hf_all.extend([T_PER_ANGLE[idx]] * len(rays_hf))

    theta_per_ray = np.array(theta_per_ray, dtype=float)
    integ_hf_all  = np.array(integ_hf_all, dtype=float)

    return rays_vhf, integ_vhf, rays_hf_all, theta_per_ray, integ_hf_all

# ===== GPU STEP 2: Dense HF sweep (angles, tracks, and T) =====
def step2_generate_rays_gpu_sweep(
    lats, alts_m,
    *,
    # angle sweep (deg from vertical)
    theta_list_hf=None,
    # satellite ground-track latitudes (deg)
    sat_lats=None,
    # per-ray polyline resolution
    npts_hf=500,
    # integration-time sweep (seconds)
    t_sweep=None,
):
    """
    Build a dense set of HF rays by sweeping angles, tracks, and integration times.
    Geometry is independent of T, so we replicate rays for each T.
    Returns:
        rays_hf        : list[np.ndarray] of shape (n_points, 2) = [lat_deg, alt_m]
        theta_per_ray  : (N,) array of true incidence angles (deg from vertical)
        integ_hf       : (N,) array of per-ray integration times (s)
    """
    import numpy as np
    from ray_trace_passive import trace_passive_oblique

    # defaults that match your existing setup if not provided
    if theta_list_hf is None:
        theta_list_hf = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
    if sat_lats is None:
        sat_lats = np.linspace(-6.0, 6.0, 181)  # denser than 60 to broaden TEC coverage
    if t_sweep is None:
        t_sweep = np.linspace(0.05, 0.15, 21)   # dense x-axis

    # ----- Build base geometry once (independent of T) -----
    rays_hf_base = []
    theta_base   = []
    for theta in theta_list_hf:
        # rays_this_theta = trace_passive_oblique(
        #     sat_lats=sat_lats,
        #     h0=alts_m.max(),
        #     hsat=alts_m.max(),
        #     theta_i_deg=theta,
        #     npts=npts_hf
        # )
        rays_this_theta = trace_passive_oblique(
            sat_lats=sat_lats,
            h0_m=alts_m.max(),
            hsat_m=alts_m.max(),
            theta_i_deg=theta,
            npts=npts_hf,
            R_E_m=1.5608e6,
            mode=RAY_TRACE_MODE,
            lat_left_deg=0.0          # << choose your left start latitude
        )
        rays_hf_base.extend(rays_this_theta)
        theta_base.extend([theta]*len(rays_this_theta))

    rays_hf = []
    theta_per_ray = []
    integ_hf = []

    # ----- Replicate geometry across the T grid -----
    for T in t_sweep:
        rays_hf.extend(rays_hf_base)
        theta_per_ray.extend(theta_base)
        integ_hf.extend([T]*len(rays_hf_base))

    return rays_hf, np.asarray(theta_per_ray, float), np.asarray(integ_hf, float)


# ---------------------- Step 3: Forward model -------------------
def step3_forward_model(iono, lats, alts_m,
                        rays_vhf, integ_vhf,
                        rays_hf,  theta_per_ray, integ_hf):
    # VHF: STEC → Δt (with noise) → VTEC (verticalized inside helper)
    stec_vhf = np.array([
        compute_STEC_along_path(iono, lats, alts_m, ray) for ray in rays_vhf
    ])
    vtec_vhf = []
    delta_t_vhf = []
    for i, ray_stec in enumerate(stec_vhf):
        vtec_i, dt_i = dualfreq_to_VTEC(
            stec_slant=np.array([ray_stec]),
            f_c=F_C_VHF,
            bw=BW_VHF,
            theta_deg=0.0,
            integration_time=integ_vhf[i],
            return_deltat=True
        )
        vtec_vhf.append(vtec_i[0])
        delta_t_vhf.append(dt_i[0])
    vtec_vhf = np.array(vtec_vhf)
    delta_t_vhf = np.array(delta_t_vhf)

    # HF: per-ray STEC → Δt → VTEC
    stec_hf = np.array([
        compute_STEC_along_path(iono, lats, alts_m, ray) for ray in rays_hf
    ])
    vtec_hf, delta_t_hf = dualfreq_to_VTEC(
        stec_slant=stec_hf,
        f_c=F_C_HF,
        bw=BW_HF,
        theta_deg=theta_per_ray,          # vectorized; helper accepts array
        integration_time=integ_hf,
        return_deltat=True
    )

    return vtec_vhf, delta_t_vhf, vtec_hf, delta_t_hf

# ===== GPU STEP 3: Forward model (TEC truth via D_hf @ x on GPU) =====
USE_GPU = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    USE_GPU = False
    cp = None
    cpx_sparse = None

def step3_forward_model_gpu(
    iono, lats, alts_m,
    D_hf,                 # geometry matrix for HF rays (SciPy CSR or dense np.ndarray)
    theta_per_ray,        # (N,) deg from vertical
    integ_hf,             # (N,) seconds
    f_c_hf=9e6, bw_hf=1e6,
):
    """
    Compute per-ray slant TEC (truth) on the GPU, then convert to VTEC & Δt
    using your existing CPU helper. Heavy part (D @ x) is GPU-accelerated.

    Returns:
        tec_true  : (N,) slant TEC truth (m^-2)
        vtec_hf   : (N,) verticalized TEC estimates (as your helper outputs)
        delta_t_hf: (N,) dual-frequency group delay with noise (s)
    """
    import numpy as np
    from ray_trace_passive import dualfreq_to_VTEC

    if not USE_GPU:
        raise RuntimeError("CuPy not available; use the CPU path or set USE_GPU=True.")

    # ---- Push ionosphere to GPU and vectorize ----
    iono_gpu = cp.asarray(iono, dtype=cp.float32)          # (n_alt, n_lat)
    x_gpu = iono_gpu.ravel(order="C")                      # (n_pix,)

    # ---- Geometry to GPU (sparse preferred) ----
    if hasattr(D_hf, "tocsr"):  # SciPy sparse
        D_hf_gpu = cpx_sparse.csr_matrix(D_hf)
    else:
        # assume dense np.ndarray
        D_hf_gpu = cpx_sparse.csr_matrix(cp.asarray(D_hf))

    # ---- TEC truth: one shot on GPU ----
    tec_true_gpu = D_hf_gpu @ x_gpu                        # (N,)
    tec_true = cp.asnumpy(tec_true_gpu)                    # back to CPU for helper

    # ---- Convert to VTEC & Δt via your existing helper (vectorized & cheap) ----
    vtec_hf, delta_t_hf = dualfreq_to_VTEC(
        stec_slant=tec_true,
        f_c=f_c_hf,
        bw=bw_hf,
        theta_deg=np.asarray(theta_per_ray, float),
        integration_time=np.asarray(integ_hf, float),
        return_deltat=True
    )
    return tec_true, vtec_hf, delta_t_hf


# ---------------------- Step 4: Build geometry ------------------
def step4_build_geometry(rays_vhf, rays_hf, lats, alts_m):
    # ALWAYS sparse + float32 to avoid OOM
    D_vhf = build_geometry_matrix_weighted_sparse(rays_vhf, lats, alts_m, dtype=np.float32)
    D_hf  = build_geometry_matrix_weighted_sparse(rays_hf,  lats, alts_m, dtype=np.float32)
    return D_vhf, D_hf

# ---------------------- Step 5: Reconstruct ---------------------
def step5_reconstruct(D_vhf, D_hf, vtec_vhf, vtec_hf,
                      delta_t_vhf, delta_t_hf,
                      lats, alts_m,
                      n_iters=N_ITERS, relax=RELAX):
    
    n_lat, n_alt = len(lats), len(alts_m)
    # 5A) ART (VHF-only, HF-only, stacked)
    # VHF-only
    if sp.issparse(D_vhf):
        Ne_rec_vhf = reconstruct_art_sparse(D_vhf, vtec_vhf, n_lat, n_alt, n_iters, relax)
    else:
        Ne_rec_vhf = reconstruct_art(D_vhf, vtec_vhf, n_lat, n_alt, n_iters, relax)

    # HF-only
    if sp.issparse(D_hf):
        Ne_rec_hf = reconstruct_art_sparse(D_hf, vtec_hf, n_lat, n_alt, n_iters, relax)
    else:
        Ne_rec_hf = reconstruct_art(D_hf, vtec_hf, n_lat, n_alt, n_iters, relax)

    # Combined
    D_all = sp.vstack([D_vhf, D_hf], format="csr") if (sp.issparse(D_vhf) or sp.issparse(D_hf)) \
            else np.vstack([D_vhf, D_hf])
    vtec_all = np.concatenate([vtec_vhf, vtec_hf])

    if sp.issparse(D_all):
        Ne_rec_all = reconstruct_art_sparse(D_all, vtec_all, n_lat, n_alt, n_iters, relax)
    else:
        Ne_rec_all = reconstruct_art(D_all, vtec_all, n_lat, n_alt, n_iters, relax)


    # 5B) Weighted ART (weights ∝ √(BW·T·SNR); here SNR assumed 1 unless you pass it)
    integ_all = np.concatenate([
        np.full_like(vtec_vhf, INTEG_TIMES_VHF, dtype=float),
        # NOTE: we don't have the per-ray HF T here yet; caller passes it:
        # We'll return a closure result instead (see wrapper below)
        # or you can pass integ_hf externally into this function.
    ])

    # We'll build weights in the wrapper where we have HF T.
    return Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, D_all, vtec_all

def step5_weighted_reconstruct(D_all, vtec_all, integ_vhf, integ_hf, bw_vhf=BW_VHF, bw_hf=BW_HF,
                               n_lat=None, n_alt=None,
                               n_iters=50, relax=0.2):
    # Build bandwidth & integration arrays aligned with stacked measurements
    all_integration_times = np.concatenate([integ_vhf, integ_hf])
    all_bandwidths = np.concatenate([
        np.full(integ_vhf.shape[0], bw_vhf),
        np.full(integ_hf.shape[0],  bw_hf)
    ])
    weights = calculate_measurement_weights(
        integration_times=all_integration_times,
        bandwidths=all_bandwidths
    )
    if sp.issparse(D_all):
        Ne_rec_weighted = weighted_reconstruction_art_sparse(
            D_all, vtec_all, weights, n_lat, n_alt, n_iters=n_iters, relax=relax
        )
    else:
        Ne_rec_weighted = weighted_reconstruction_art(
            D_all, vtec_all, weights, n_lat, n_alt, n_iters=n_iters, relax=relax
        )
    return Ne_rec_weighted, weights

# ---------------------- Step 6: Evaluate & Plot -----------------
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b)**2)))

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0, b0 = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    return float((a0 * b0).sum() / denom) if denom > 0 else 0.0

def plot_recons(lats, alts_m, recs, titles):
    fig, axs = plt.subplots(1, len(recs), figsize=(6*len(recs), 6), sharey=True)
    if len(recs) == 1: axs = [axs]
    for ax, rec, title in zip(axs, recs, titles):
        x_edges = centers_to_edges(lats)
        y_edges = centers_to_edges(alts_m/1e3)
        # Mask zeros so background is transparent
        rec_masked = np.where(rec == 0, np.nan, rec)
        im = ax.pcolormesh(x_edges, y_edges, rec_masked*1e-6, shading='auto', cmap='viridis')

        ax.set_title(title)
        ax.set_xlabel("Latitude (°)")
        if ax is axs[0]:
            ax.set_ylabel("Altitude (km)")
        fig.colorbar(im, ax=ax, label='Ne (×10⁶ cm⁻³)')
    plt.tight_layout(); plt.show(block=False); plt.savefig(os.path.join(IMG_DIR, "A_gpu_reconstructions.png"))

def plot_raypaths(lats, alts_m, rays_hf, theta_per_ray, rays_vhf,
                  title_hf="Modeled Raypaths (HF)",
                  title_vhf="Modeled Raypaths (VHF)",
                  max_rays_per_angle=None):
    """
    Two-panel figure:
      left  = HF oblique rays, color-coded by true incidence angle (deg from vertical)
      right = VHF nadir rays (vertical)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    # group HF rays by angle (preserve order)
    angle_to_indices = defaultdict(list)
    for i, th in enumerate(theta_per_ray):
        angle_to_indices[float(th)].append(i)

    # Visually nice, ordered by increasing angle (like your example)
    angles_sorted = sorted(angle_to_indices.keys())

    # Pick distinct colors for angles (same order every time)
    # You can tweak the list to match your house style
    palette = [
        "#0015BC",  # deep blue
        "#4527A0",  # indigo
        "#8E24AA",  # purple
        "#D81B60",  # magenta
        "#FB8C00",  # orange
        "#FFC107",  # amber
        "#FDD835",  # yellow
        "#43A047",  # green (backup if needed)
    ]
    # map angle -> color (wrap if more angles)
    angle_color = {ang: palette[i % len(palette)] for i, ang in enumerate(angles_sorted)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    ax_hf, ax_vhf = axes

    # ---- HF panel ----
    for ang in angles_sorted:
        idxs = angle_to_indices[ang]
        if max_rays_per_angle is not None:
            idxs = idxs[:max_rays_per_angle]
        col = angle_color[ang]
        for i in idxs:
            ray = rays_hf[i]
            # ray: (n_points, 2) = [lat_deg, alt_m]
            ax_hf.plot(ray[:, 0], ray[:, 1] / 1e3, lw=0.6, color=col, alpha=0.9)

    # Legend — one entry per angle
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=angle_color[ang], lw=2, label=f"{int(round(ang))}°")
        for ang in angles_sorted
    ]
    leg = ax_hf.legend(
        handles=legend_handles,
        title="HF θ (deg, from vertical)",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        framealpha=0.9,
        fontsize=9,
        title_fontsize=10,
        ncol=1
    )

    ax_hf.set_title(title_hf)
    ax_hf.set_xlabel("Latitude (Degree)")
    ax_hf.set_ylabel("Altitude (km)")
    ax_hf.set_xlim(lats.min(), lats.max())
    ax_hf.set_ylim(0, alts_m.max() / 1e3)
    ax_hf.grid(False)

    # ---- VHF panel ----
    for ray in rays_vhf:
        ax_vhf.plot(ray[:, 0], ray[:, 1] / 1e3, lw=0.6, color="#5C6BC0", alpha=0.9)

    ax_vhf.set_title(title_vhf)
    ax_vhf.set_xlabel("Latitude (Degree)")
    ax_vhf.set_xlim(lats.min(), lats.max())
    ax_vhf.set_ylim(0, alts_m.max() / 1e3)
    ax_vhf.grid(False)

    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(os.path.join(IMG_DIR, "B_gpu_raypaths.png"))

def build_passive_plot_inputs(
    iono, lats, alts_m,
    rays_hf,                  # list of HF polylines (n_hf_rays)
    integ_hf,                 # (n_hf_rays,) integration times [s]
    theta_per_ray,            # (n_hf_rays,) true incidence angle from vertical [deg]
    vtec_hf,                  # (n_hf_rays,) VTEC estimates returned by your dualfreq helper
    delta_t_hf,               # (n_hf_rays,) noisy HF dual-freq delay [s]
    delta_t_vhf,              # (n_vhf_rays,) noisy VHF dual-freq delay [s]
    sat_lats_vhf=np.linspace(-6.0, 6.0, 60),
    hf_origin_lat=None,   # must match how you built VHF rays
    F_C_HF=9e6, BW_HF=1e6,                 # HF center & bandwidth [Hz]
    F_C_VHF=60e6,                          # VHF center [Hz]
):
    # 1) x-axis: per-HF-ray integration time (seconds)
    t_meas = integ_hf.astype(float)

    # 2) y-axis: true slant TEC per HF ray (m^-2)
    tec_for_bin = np.array([compute_STEC_along_path(iono, lats, alts_m, ray)
                            for ray in rays_hf], dtype=float)

    # 3A) Passive TEC estimator absolute error (m^-2)
    theta_rad = np.deg2rad(theta_per_ray)
    stec_hf_est = vtec_hf / np.cos(theta_rad)           # estimated slant TEC (m^-2)
    delta_tec_err_passive = np.abs(stec_hf_est - tec_for_bin)

    # 3B) Residual TEC after passive HF correction via δ(Δt) → VHF range error (m)
    f1_h = F_C_HF - 0.5 * BW_HF
    f2_h = F_C_HF + 0.5 * BW_HF
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    # Match each HF ray to a single VHF reference Δt (same convention as your delta_dt_branch)
    if hf_origin_lat is None:
        hf_origin_lat = float(sat_lats_vhf[0])
    vhf_idx = int(np.argmin(np.abs(np.asarray(sat_lats_vhf) - hf_origin_lat)))
    delta_t_vhf_matched = np.full_like(delta_t_hf, delta_t_vhf[vhf_idx])

    delta_dt = -(delta_t_vhf_matched - delta_t_hf)                  # seconds
    tec_residual_after_passive = (C * delta_dt / K_IONO) * freq_factor_h   # m^-2
    range_error_after_passive = (K_IONO / (F_C_VHF**2)) * np.abs(tec_residual_after_passive)  # meters

    return t_meas, tec_for_bin, delta_tec_err_passive, range_error_after_passive


def plot_passive_tec_error_heatmap_from_arrays(
    t_meas, tec_for_bin, delta_tec_err_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50,
    title="Average TEC Estimate Error for Passive (Simulated)",
    cmap="plasma",
    smooth_sigma: float = 0.0,
):
    df = pd.DataFrame({
        "vtec": tec_for_bin,           # y-axis (slant TEC truth)
        "t": t_meas,                   # x-axis (integration time, s)
        "err": delta_tec_err_passive,  # color (|ΔTEC|, m^-2)
    })

    # Filter to match your original figure window
    df = df[(df['t'] >= t_min) & (df['t'] <= t_max)]
    df = df[(df['vtec'] >= tec_min) & (df['vtec'] <= tec_max)]

    tec_bins = np.linspace(tec_min, tec_max, nbins_tec)
    t_bins   = np.linspace(t_min,   t_max,   nbins_t)

    heatmap, yedges, xedges = np.histogram2d(
        df['vtec'], df['t'], bins=[tec_bins, t_bins], weights=df['err'])
    counts,  _,     _       = np.histogram2d(
        df['vtec'], df['t'], bins=[tec_bins, t_bins])

    # Compute average (safe divide)
    avg_error = heatmap / np.maximum(counts, 1)

    # Default to top-level SMOOTH_SIGMA if caller omitted
    if smooth_sigma == 0.0:
        smooth_sigma = SMOOTH_SIGMA

    # Optional smoothing: smooth numerator & denominator separately to avoid bias
    if smooth_sigma and smooth_sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
            heat_s = gaussian_filter(heatmap, sigma=smooth_sigma)
            counts_s = gaussian_filter(counts.astype(float), sigma=smooth_sigma)
            avg_error = heat_s / np.maximum(counts_s, 1e-8)
        except Exception:
            # scipy not available; continue without smoothing
            pass

    # Mask bins with zero raw counts so they don't dominate the colormap
    avg_display = np.ma.array(avg_error, mask=(counts == 0))

    plt.figure(figsize=(8,6))
    plt.pcolormesh(xedges, yedges, avg_display, shading='auto', cmap=cmap)
    plt.xlabel("Integration Time (s)")
    plt.ylabel("TEC (electrons/m$^{2}$)")
    plt.title(title)
    plt.colorbar(label="ΔTEC Error (m$^{-2}$)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(os.path.join(IMG_DIR, "C_passive_tec_error_heatmap_from_arrays.png"))


def plot_vhf_range_error_heatmap_from_arrays(
    t_meas, tec_for_bin, range_error_after_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50,
    title="VHF Range Error after Passive HF Correction",
    cmap="jet",
    smooth_sigma: float = 0.0,
):
    df_range = pd.DataFrame({
        "vtec": tec_for_bin,                    # y-axis (slant TEC truth)
        "t": t_meas,                            # x-axis (integration time, s)
        "range_err": range_error_after_passive  # color (meters)
    })

    df_range = df_range[(df_range['t'] >= t_min) & (df_range['t'] <= t_max)]
    df_range = df_range[(df_range['vtec'] >= tec_min) & (df_range['vtec'] <= tec_max)]

    tec_bins = np.linspace(tec_min, tec_max, nbins_tec)
    t_bins   = np.linspace(t_min,   t_max,   nbins_t)

    heatmap_r, yedges_r, xedges_r = np.histogram2d(
        df_range['vtec'], df_range['t'],
        bins=[tec_bins, t_bins], weights=df_range['range_err'])
    counts_r, _, _ = np.histogram2d(
        df_range['vtec'], df_range['t'], bins=[tec_bins, t_bins])

    avg_range_err = heatmap_r / np.maximum(counts_r, 1)

    if smooth_sigma == 0.0:
        smooth_sigma = SMOOTH_SIGMA

    if smooth_sigma and smooth_sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
            heat_s = gaussian_filter(heatmap_r, sigma=smooth_sigma)
            counts_s = gaussian_filter(counts_r.astype(float), sigma=smooth_sigma)
            avg_range_err = heat_s / np.maximum(counts_s, 1e-8)
        except Exception:
            pass

    avg_display = np.ma.array(avg_range_err, mask=(counts_r == 0))

    plt.figure(figsize=(8,6))
    plt.pcolormesh(xedges_r, yedges_r, avg_display, shading='auto', cmap=cmap)
    plt.xlabel("Integration Time (sec)")
    plt.ylabel("Total Electron Content (m$^{-2}$)")
    plt.title(title)
    plt.colorbar(label="Δr (meters)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(os.path.join(IMG_DIR, "D_vhf_range_error_heatmap.png"))


def plot_passive_tec_error_heatmap_GPU(
    t_meas, tec_for_bin, delta_tec_err_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50, title="Average TEC Estimate Error for Passive (Simulated)",
    smooth_sigma: float = 0.0,
):
    # Filter on CPU (cheap) to keep plotting logic identical
    df = pd.DataFrame({"t": t_meas, "vtec": tec_for_bin, "err": delta_tec_err_passive})
    df = df[(df["t"]>=t_min)&(df["t"]<=t_max)&(df["vtec"]>=tec_min)&(df["vtec"]<=tec_max)]

    t_edges   = np.linspace(t_min,   t_max,   nbins_t)
    tec_edges = np.linspace(tec_min, tec_max, nbins_tec)

    H, C = _gpu_binned_mean_2d(df["t"].values, df["vtec"].values, df["err"].values, t_edges, tec_edges)

    if smooth_sigma == 0.0:
        smooth_sigma = SMOOTH_SIGMA

    # Default: compute average safely
    avg_error = H / np.maximum(C, 1)

    if smooth_sigma and smooth_sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
            Hs = gaussian_filter(H, sigma=smooth_sigma)
            Cs = gaussian_filter(C.astype(float), sigma=smooth_sigma)
            avg_error = Hs / np.maximum(Cs, 1e-8)
        except Exception:
            pass

    avg_display = np.ma.array(avg_error, mask=(C == 0))

    plt.figure(figsize=(8,6))
    plt.pcolormesh(t_edges, tec_edges, avg_display, shading='auto', cmap="plasma")
    plt.xlabel("Integration Time (s)")
    plt.ylabel("TEC (electrons/m$^{2}$)")
    plt.title(title)
    plt.colorbar(label="ΔTEC Error (m$^{-2}$)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(os.path.join(IMG_DIR, "C_gpu_hf_passive_tec_error_heatmap.png"))


def plot_vhf_range_error_heatmap_GPU(
    t_meas, tec_for_bin, range_error_after_passive,
    t_min=0.05, t_max=0.15, tec_min=1.5e15, tec_max=5e15,
    nbins_t=50, nbins_tec=50, title="VHF Range Error after Passive HF Correction",
    smooth_sigma: float = 0.0,
):
    import pandas as pd
    df = pd.DataFrame({"t": t_meas, "vtec": tec_for_bin, "rng": range_error_after_passive})
    df = df[(df["t"]>=t_min)&(df["t"]<=t_max)&(df["vtec"]>=tec_min)&(df["vtec"]<=tec_max)]

    t_edges   = np.linspace(t_min,   t_max,   nbins_t)
    tec_edges = np.linspace(tec_min, tec_max, nbins_tec)

    H, C = _gpu_binned_mean_2d(df["t"].values, df["vtec"].values, df["rng"].values, t_edges, tec_edges)

    if smooth_sigma == 0.0:
        smooth_sigma = SMOOTH_SIGMA

    avg_range_err = H / np.maximum(C, 1)
    if smooth_sigma and smooth_sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
            Hs = gaussian_filter(H, sigma=smooth_sigma)
            Cs = gaussian_filter(C.astype(float), sigma=smooth_sigma)
            avg_range_err = Hs / np.maximum(Cs, 1e-8)
        except Exception:
            pass

    avg_display = np.ma.array(avg_range_err, mask=(C == 0))

    plt.figure(figsize=(8,6))
    plt.pcolormesh(t_edges, tec_edges, avg_display, shading='auto', cmap="jet")
    plt.xlabel("Integration Time (sec)")
    plt.ylabel("Total Electron Content (m$^{-2}$)")
    plt.title(title)
    plt.colorbar(label="Δr (meters)")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(os.path.join(IMG_DIR, "D_gpu_vhf_range_error_heatmap.png"))


def _colcounts_to_grid(col_counts, n_alt, n_lat):
    """Reshape per-voxel column counts (length n_alt*n_lat) to (n_alt, n_lat) heatmap."""
    return np.asarray(col_counts, float).reshape(n_alt, n_lat, order="C")

def _imshow_lat_alt(img_alt_lat, lats, alts_m, title, cmap="viridis", cbar_label=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(img_alt_lat, origin="lower", aspect="auto",
                   extent=[lats.min(), lats.max(), alts_m.min()/1e3, alts_m.max()/1e3],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Latitude (deg)"); ax.set_ylabel("Altitude (km)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig, ax

# 1) Ray coverage heatmap (per-voxel hit counts)
def plot_ray_coverage_heatmap(D, lats, alts_m, title="Ray coverage (hits per voxel)"):
    """
    D: CSR geometry (n_rays x (n_alt*n_lat)).
    Shows, for each voxel, number of rays that intersect it.
    """
    n_alt, n_lat = len(alts_m), len(lats)
    col_counts = D.getnnz(axis=0) if sp.issparse(D) else np.count_nonzero(D, axis=0)
    grid = _colcounts_to_grid(col_counts, n_alt, n_lat)
    fig, ax = _imshow_lat_alt(grid, lats, alts_m, title, cmap="magma", cbar_label="# ray hits")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-17, 1e-13)
    plt.savefig(os.path.join(IMG_DIR, "E_ray_coverage_heatmap.png"))
    return fig, ax

# 2) Δt vs 1/f² check
def plot_dt_vs_invf2(delta_t_hf, delta_t_vhf, f_hf, f_vhf, title="|Δt| vs 1/f²"):
    # x = 1/f^2 for each band (constant per band)
    invf2_hf  = np.full_like(delta_t_hf,  1.0 / (f_hf**2),  dtype=float)
    invf2_vhf = np.full_like(delta_t_vhf, 1.0 / (f_vhf**2), dtype=float)

    y_hf  = np.abs(delta_t_hf)
    y_vhf = np.abs(delta_t_vhf)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(invf2_hf,  y_hf,  s=6,  alpha=0.4, label="HF")
    ax.scatter(invf2_vhf, y_vhf, s=20, alpha=0.7, label="VHF", marker="x")

    # log–log so both clusters are visible and trend is linear with slope ≈ 1
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Nice bounds
    x_all = np.concatenate([invf2_hf, invf2_vhf])
    y_all = np.concatenate([y_hf, y_vhf])
    ax.set_xlim(x_all.min()*0.8, x_all.max()*1.2)
    ax.set_ylim(max(y_all.min()*0.8, 1e-12*np.max(y_all)), y_all.max()*1.2)

    ax.set_xlabel("1 / f² (Hz⁻²)")
    ax.set_ylabel("|Δt| (s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Optional: draw a reference line with slope 1 through the HF median point
    x0 = np.median(invf2_hf)
    y0 = np.median(y_hf)
    x_ref = np.array([x_all.min(), x_all.max()])
    y_ref = y0 * (x_ref / x0)  # slope 1 in log–log means y ~ x
    ax.plot(x_ref, y_ref, ls="--", lw=1.2, color="gray", label="_nolegend_")

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "E_dt_vs_invf2.png"), dpi=200, bbox_inches="tight")
    return fig, ax

def plot_dt_vs_invf2(delta_t_hf, delta_t_vhf, f_hf, f_vhf, title="|Δt| vs 1/f²"):
    # x = 1/f^2 for each band (constant per band)
    invf2_hf  = np.full_like(delta_t_hf,  1.0 / (f_hf**2),  dtype=float)
    invf2_vhf = np.full_like(delta_t_vhf, 1.0 / (f_vhf**2), dtype=float)

    y_hf  = np.abs(delta_t_hf)
    y_vhf = np.abs(delta_t_vhf)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(invf2_hf,  y_hf,  s=6,  alpha=0.4, label="HF")
    ax.scatter(invf2_vhf, y_vhf, s=20, alpha=0.7, label="VHF", marker="x")

    # log–log so both clusters are visible and trend is linear with slope ≈ 1
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Nice bounds
    x_all = np.concatenate([invf2_hf, invf2_vhf])
    y_all = np.concatenate([y_hf, y_vhf])
    ax.set_xlim(x_all.min()*0.8, x_all.max()*1.2)
    ax.set_ylim(max(y_all.min()*0.8, 1e-12*np.max(y_all)), y_all.max()*1.2)

    ax.set_xlabel("1 / f² (Hz⁻²)")
    ax.set_ylabel("|Δt| (s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Optional: draw a reference line with slope 1 through the HF median point
    x0 = np.median(invf2_hf)
    y0 = np.median(y_hf)
    x_ref = np.array([x_all.min(), x_all.max()])
    y_ref = y0 * (x_ref / x0)  # slope 1 in log–log means y ~ x
    ax.plot(x_ref, y_ref, ls="--", lw=1.2, color="gray", label="_nolegend_")

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "E_dt_vs_invf2.png"), dpi=200, bbox_inches="tight")
    return fig, ax


# 3) VTEC recovery vs incidence angle (synthetic)
def plot_vtec_recovery_vs_angle(vtec_true, angles_deg, f0, bw, T, dualfreq_to_VTEC, title="VTEC recovery vs angle (synthetic)"):
    th = np.asarray(angles_deg, float)
    stec = vtec_true / np.cos(np.deg2rad(th))  # synthetic slant
    vtec_est, _ = dualfreq_to_VTEC(stec_slant=stec, f_c=f0, bw=bw, theta_deg=th, integration_time=np.full_like(th, T), return_deltat=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(th, vtec_est, lw=2, label="Recovered VTEC")
    ax.axhline(vtec_true, ls="--", label="True VTEC")
    ax.set_xlabel("Incidence angle θ (deg from vertical)")
    ax.set_ylabel("VTEC (m⁻²)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    #plt.savefig(os.path.join(IMG_DIR, "F_vtec_recovery_vs_angle.png"))
    return fig, ax

# 4) Residual histograms (unweighted vs weighted)
def plot_residual_histograms(D_all, vtec_all, Ne_rec_all, Ne_rec_weighted, title="Residual histograms"):
    yhat_all = D_all @ Ne_rec_all.ravel(order="C")
    yhat_w   = D_all @ Ne_rec_weighted.ravel(order="C")
    r_all = (yhat_all - vtec_all).astype(float)
    r_w   = (yhat_w   - vtec_all).astype(float)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(r_all, bins=100, alpha=0.5, label="Unweighted", density=True)
    ax.hist(r_w,   bins=100, alpha=0.5, label="Weighted",   density=True)
    ax.set_xlabel("Residual (pred - meas)"); ax.set_ylabel("Density"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "G_residual_histograms.png"))
    return fig, ax, r_all, r_w

# 5) Residual vs angle / vs integration time (HF only)
def plot_residuals_vs_angle_time(D_all, vtec_all, D_vhf, theta_per_ray_hf, integ_hf, Ne_rec, title_prefix="HF residuals"):
    """
    Assumes D_all has VHF rows first, then HF rows, so HF residual slice starts at D_vhf.shape[0].
    """
    n_vhf = D_vhf.shape[0]
    yhat = D_all @ Ne_rec.ravel(order="C")
    r = (yhat - vtec_all).astype(float)
    r_hf = r[n_vhf:]
    th = np.asarray(theta_per_ray_hf, float)
    T  = np.asarray(integ_hf, float)

    fig1, ax1 = plt.subplots(figsize=(6.6, 4.6))
    ax1.scatter(th, np.abs(r_hf), s=6, alpha=0.5)
    ax1.set_xlabel("θ (deg from vertical)")
    ax1.set_ylabel("|Residual|")
    ax1.set_title(f"{title_prefix}: |resid| vs θ"); ax1.grid(True, alpha=0.3)

    # Bin by unique T values (or quantile bins if continuous)
    unique_T = np.unique(T)
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.6))
    data = [np.abs(r_hf[T==t]) for t in unique_T]
    ax2.boxplot(data, labels=[f"{t:.2f}s" for t in unique_T], showfliers=False)
    ax2.set_xlabel("Integration time T (s)")
    ax2.set_ylabel("|Residual|")
    ax2.set_title(f"{title_prefix}: |resid| vs T"); ax2.grid(True, alpha=0.3)
    fig1.tight_layout(); fig2.tight_layout()

    fig1.savefig(os.path.join(IMG_DIR, "H_residuals_vs_angle.png"))
    fig2.savefig(os.path.join(IMG_DIR, "I_residuals_vs_integration_time.png"))
    return (fig1, ax1), (fig2, ax2)

# 6) Reconstruction cross-sections (2D & 1D)
def plot_recon_cross_sections(
    lats, alts_m,
    truth_alt_lat, rec_alt_lat_all, rec_alt_lat_w,
    lat_samples_deg=(-5, 0, 5),
    title_prefix="Reconstruction"
):
    """
    Inputs may be (n_alt, n_lat) or (n_lat, n_alt). This function will orient to (n_alt, n_lat).
    """
    n_alt, n_lat = len(alts_m), len(lats)

    def orient(A, name):
        if A.shape == (n_alt, n_lat):
            return A
        if A.shape == (n_lat, n_alt):
            return A.T
        raise ValueError(f"{name} has shape {A.shape}, expected (n_alt,{n_lat}) or (n_lat,{n_alt}).")

    truth = orient(truth_alt_lat, "truth_alt_lat")
    rec_all = orient(rec_alt_lat_all, "rec_alt_lat_all")
    rec_w   = orient(rec_alt_lat_w,   "rec_alt_lat_w")

    err_all = rec_all - truth
    err_w   = rec_w   - truth

    figs = []

    # 2D panels
    for label, img, cbar_lbl, fname in [
        ("Truth",              truth,   "Ne (m⁻³)",           "J1_truth.png"),
        ("Recon (all)",        rec_all, "Ne (m⁻³)",           "J2_recon_all.png"),
        ("Recon (weighted)",   rec_w,   "Ne (m⁻³)",           "J3_recon_weighted.png"),
        ("Error (all)",        err_all, "ΔNe (m⁻³)",          "J4_error_all.png"),
        ("Error (weighted)",   err_w,   "ΔNe (m⁻³)",          "J5_error_weighted.png"),
    ]:
        fig, ax = _imshow_lat_alt(img, lats, alts_m, f"{title_prefix} — {label}",
                                  cbar_label=cbar_lbl)
        fig.savefig(os.path.join(IMG_DIR, fname), dpi=200, bbox_inches="tight")
        figs.append((fig, ax))

    # 1D altitude profiles at chosen latitudes
    lat_idx = [int(np.argmin(np.abs(lats - v))) for v in lat_samples_deg]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for j in lat_idx:
        ax.plot(alts_m/1e3, truth[:, j],   lw=2.0, label=f"truth @ {lats[j]:.1f}°")
        ax.plot(alts_m/1e3, rec_all[:, j], lw=1.2, ls="--", label=f"all @ {lats[j]:.1f}°")
        ax.plot(alts_m/1e3, rec_w[:, j],   lw=1.2, ls=":",  label=f"weighted @ {lats[j]:.1f}°")

    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Ne (m⁻³)")
    ax.set_title(f"{title_prefix} — altitude profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e8, 1e12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "J_recon_cross_sections.png"), dpi=200, bbox_inches="tight")
    figs.append((fig, ax))

    return figs

# 7) RMSE vs iteration (ART)
def plot_rmse_vs_iteration(rmse_list_unweighted, rmse_list_weighted=None, title="ART convergence"):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(np.arange(1, len(rmse_list_unweighted)+1), rmse_list_unweighted, lw=2, label="Unweighted")
    if rmse_list_weighted is not None:
        ax.plot(np.arange(1, len(rmse_list_weighted)+1), rmse_list_weighted, lw=2, label="Weighted")
    ax.set_xlabel("Iteration"); ax.set_ylabel("RMSE (pred - meas)")
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); 
    plt.savefig(os.path.join(IMG_DIR, "J_rmse_vs_iteration.png"))
    return fig, ax

# 8) L-curve (optional)
def plot_lcurve(norm_residual, norm_regularizer, title="L-curve"):
    """
    norm_residual[k] = ||D·Ne_k - V||_2, norm_regularizer[k] = ||L·Ne_k||_2 (you compute L if you use it).
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.loglog(norm_residual, norm_regularizer, marker="o")
    ax.set_xlabel("||D·Ne - V||₂"); ax.set_ylabel("||L·Ne||₂"); ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); 
    plt.savefig(os.path.join(IMG_DIR, "K_lcurve.png"))
    return fig, ax

# 9) Range error after passive correction (your arrays)
def plot_range_error_heatmaps(t_meas, tec_for_bin, delta_tec_err_passive, range_error_after_passive,
                              title_prefix="Passive correction performance"):
    """
    t_meas, tec_for_bin define a 2D grid (like np.meshgrid) for heatmaps.
    Arrays delta_tec_err_passive & range_error_after_passive must match that grid.
    """
    fig1, ax1 = plt.subplots(figsize=(6.8, 4.6))
    im1 = ax1.pcolormesh(t_meas, tec_for_bin, delta_tec_err_passive, shading="auto")
    ax1.set_xlabel("Integration time (s)"); ax1.set_ylabel("Measured TEC bin")
    ax1.set_title(f"{title_prefix}: ΔTEC error (after HF)"); plt.colorbar(im1, ax=ax1, label="ΔTEC (m⁻²)")

    fig2, ax2 = plt.subplots(figsize=(6.8, 4.6))
    im2 = ax2.pcolormesh(t_meas, tec_for_bin, range_error_after_passive, shading="auto")
    ax2.set_xlabel("Integration time (s)"); ax2.set_ylabel("Measured TEC bin")
    ax2.set_title(f"{title_prefix}: VHF range error (m)"); plt.colorbar(im2, ax=ax2, label="Range error (m)")
    for fig in (fig1, fig2):
        fig.tight_layout()

    fig1.savefig(os.path.join(IMG_DIR, "L_range_error_heatmap_delta_tec.png"))
    fig2.savefig(os.path.join(IMG_DIR, "L_range_error_heatmap_range_error.png"))
    return (fig1, ax1), (fig2, ax2)

# 10) Per-voxel uncertainty proxy (DᵀD diagonal approx)
def plot_voxel_uncertainty_proxy(D_all, lats, alts_m, title="Per-voxel uncertainty proxy"):
    """
    Use diag(DᵀD) ≈ sum over rays of weight² for each voxel (cheap proxy for variance^-1).
    """
    n_alt, n_lat = len(alts_m), len(lats)
    if sp.issparse(D_all):
        # sum of squared entries per column
        D2 = D_all.copy()
        D2.data = D2.data**2
        diag_proxy = np.array(D2.sum(axis=0)).ravel()
    else:
        diag_proxy = np.sum(D_all*D_all, axis=0)
    grid = _colcounts_to_grid(diag_proxy, n_alt, n_lat)
    fig = _imshow_lat_alt(grid, lats, alts_m, title, cmap="plasma", cbar_label="∑ weights² (arb)")
    vmax = np.percentile(diag_proxy, 99)
    plt.imshow(grid, norm=colors.LogNorm(vmin=1e8, vmax=vmax), cmap="plasma")
    plt.savefig(os.path.join(IMG_DIR, "M_voxel_uncertainty_proxy.png"))
    return fig

def _rmse_from(D, x_c, y):
    """RMSE of residuals r = D x - y."""
    if sp.issparse(D):
        r = D @ x_c
    else:
        r = D.dot(x_c)
    r -= y
    return float(np.sqrt(np.mean((r)**2)))

def sirt_cimmino_with_history(D, y, *, n_lat, n_alt, n_iters=20, relax=0.1, x0=None):
    """
    Simultaneous ART (Cimmino). One iteration:
        r  = y - D x
        rn = r / ||row||   (row-wise normalization)
        x += relax * D^T rn
    Stores RMSE each iteration.
    """
    m, n = D.shape
    x = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, float).copy()

    if sp.issparse(D):
        row_norm2 = np.array(D.power(2).sum(axis=1)).ravel()
    else:
        row_norm2 = np.sum(D*D, axis=1)
    row_norm2[row_norm2 == 0.0] = 1.0

    rmse_hist = []
    for _ in range(n_iters):
        # residual
        r = (y - (D @ x)) if sp.issparse(D) else (y - D.dot(x))
        rn = r / row_norm2
        # backprojection
        bp = (D.T @ rn) if sp.issparse(D) else D.T.dot(rn)
        x += relax * bp
        # diagnostics
        err = (D @ x) - y if sp.issparse(D) else D.dot(x) - y
        rmse_hist.append(float(np.sqrt(np.mean(err**2))))
    return x.reshape(n_alt, n_lat, order="C"), rmse_hist

def art_with_history(D, y, *, n_lat, n_alt, n_iters=20, relax=0.1, x0=None,
                     compute_lcurve=False, L=None):
    """
    Classic Kaczmarz ART across all rows, for `n_iters` sweeps.
    Records RMSE after each full sweep.
    Optional L-curve: store ||x||_2 (or ||Lx||_2 if L provided).
    Returns: x_map (n_alt,n_lat), rmse_list, reg_norm_list (or None)
    """
    m, n = D.shape
    x = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, float).copy()

    # precompute row norms for efficiency
    # if sp.issparse(D):
    #     row_norm2 = np.array(D.power(2).sum(axis=1)).ravel()
    # else:
    #     row_norm2 = np.sum(D*D, axis=1)
    # row_norm2[row_norm2 == 0.0] = 1.0
    # if sp.issparse(D):
    #     # Fast CSR row access (no getrow!)
    #     start = D.indptr[i]
    #     end   = D.indptr[i+1]
    #     idx   = D.indices[start:end]
    #     vals  = D.data[start:end]
    #     pred_i = np.dot(vals, x[idx])              # scalar
    #     ri     = y[i] - pred_i
    #     alpha  = relax * ri / row_norm2[i]
    #     x[idx] += alpha * vals                      # in-place update
    # else:
    #     Di     = D[i, :]
    #     pred_i = float(np.dot(Di, x))
    #     ri     = y[i] - pred_i
    #     alpha  = relax * ri / row_norm2[i]
    #     x     += alpha * Di

    # Ensure fast CSR layout for row-wise Kaczmarz
    if sp.issparse(D) and not sp.isspmatrix_csr(D):
        D = D.tocsr()

    if sp.issparse(D):
        row_norm2 = np.array(D.power(2).sum(axis=1)).ravel()
    else:
        row_norm2 = np.sum(D * D, axis=1)
    row_norm2[row_norm2 == 0.0] = 1.0

    rmse_hist, reg_hist = [], []

    for it in range(n_iters):
        # one full sweep
        # for i in range(m):
        #     # r_i = y_i - <D_i, x>
        #     Di = D.getrow(i) if sp.issparse(D) else D[i, :]
        #     pred_i = float(Di.dot(x)) if sp.issparse(D) else float(np.dot(Di, x))
        #     ri = y[i] - pred_i
        #     alpha = relax * ri / row_norm2[i]
        #     if sp.issparse(D):
        #         x[Di.indices] += alpha * Di.data
        #     else:
        #         x += alpha * Di
        for i in range(m):
            if sp.issparse(D):
                start = D.indptr[i]
                end   = D.indptr[i+1]
                idx   = D.indices[start:end]
                vals  = D.data[start:end]
                pred_i = np.dot(vals, x[idx])
                ri     = y[i] - pred_i
                alpha  = relax * ri / row_norm2[i]
                x[idx] += alpha * vals
            else:
                Di     = D[i, :]
                pred_i = float(np.dot(Di, x))
                ri     = y[i] - pred_i
                alpha  = relax * ri / row_norm2[i]
                x     += alpha * Di
            # Di = D.getrow(i) if sp.issparse(D) else D[i, :]
            # pred_i = float(Di.dot(x)) if sp.issparse(D) else float(np.dot(Di, x))
            # ri = y[i] - pred_i
            # alpha = relax * ri / row_norm2[i]
            # if sp.issparse(D):
            #     # x += alpha * D_i^T
            #     x[Di.indices] += alpha * Di.data
            # else:
            #     x += alpha * Di

        # end of sweep: diagnostics
        rmse_hist.append(_rmse_from(D, x, y))
        if compute_lcurve:
            if L is None:
                reg_hist.append(float(np.linalg.norm(x)))
            else:
                if sp.issparse(L):
                    reg_hist.append(float(np.linalg.norm(L @ x)))
                else:
                    reg_hist.append(float(np.linalg.norm(L.dot(x))))

    return x.reshape(n_alt, n_lat, order="C"), rmse_hist, (reg_hist if compute_lcurve else None)

def weighted_art_with_history(D, y, w, *, n_lat, n_alt, n_iters=20, relax=0.1, x0=None,
                              compute_lcurve=False, L=None):
    """
    Weighted ART variant: solves W^(1/2) D x ≈ W^(1/2) y.
    Weight vector w is per-row inverse-variance proxy (>=0).
    """
    w = np.asarray(w, float)
    sw = np.sqrt(np.maximum(w, 0.0))

    if sp.issparse(D):
        # ensure CSR before scaling
        D = D.tocsr() if not sp.isspmatrix_csr(D) else D
        Ds = D.multiply(sw[:, None])
        # keep CSR after multiply
        if not sp.isspmatrix_csr(Ds):
            Ds = Ds.tocsr()
    else:
        Ds = D * sw[:, None]

    ys = y * sw

    return art_with_history(Ds, ys, n_lat=n_lat, n_alt=n_alt,
                            n_iters=n_iters, relax=relax, x0=x0,
                            compute_lcurve=compute_lcurve, L=L)

# ---------------------- Step7: Peters MATLAB validation -----------------
# ---------- Peters Fig. 7 (ellipse toy model) ----------

def peters_dtec_from_range(dr=3.0, fv=60e6, fh=9e6, alpha=1.69e-6, c=3e8):
    """
    MATLAB lines:
        dTEC = (2*dr*2*pi*fv^2)/(c*alpha)
    """
    return (2*dr*2*np.pi*fv*fv)/(c*alpha)

def peters_dt_from_TEC(TEC, fv=60e6, fh=9e6, c=3e8):
    """
    MATLAB lines:
        dt_true = (TEC*80.6)*((fv^2)-(fh^2))/(c*(fv^2)*(fh^2))
    80.6 = c * K_ionosphere in SI units (≈ 40.3*c/ (2π) depending on convention).
    We mirror MATLAB so you can 1:1 reproduce its numbers.
    """
    return (TEC*80.6)*((fv**2)-(fh**2))/(c*(fv**2)*(fh**2))

def peters_theta_tecerror_surface(e_vec=None, ra=4e15, n_x=1000):
    """
    Recreates the core of the MATLAB loop that fills TEC_error(e, theta).
    Returns:
        thetas_deg: 1D array (sorted 0..~84 deg)
        e_vec     : 1D array of eccentricities
        TEC_err   : 2D array shape (len(e_vec), len(thetas_deg))
    Notes:
      - We follow the MATLAB derivation literally:
            f = ra*e, rb = sqrt(ra^2 - f^2)
            xe = linspace(-ra, f, n_x)
            ye = sqrt(rb^2*(1 - (xe^2)/(ra^2)))
            theta = atan(|xe - f| / ye)   (radians)
            yf = sqrt(rb^2*((f^2)/(ra^2) - 1))   # may be imaginary; use abs()
            two_wayTEC = 2*|yf|
            ellipseTEC = |yf| + sqrt(ye^2 + (xe - f)^2)
            TEC_error = ellipseTEC - two_wayTEC
      - Then we sort by theta so columns correspond to monotonically increasing incidence angle.
    """
    if e_vec is None:
        e_vec = np.linspace(0.01, 0.99, 200)   # matches spirit of paper

    thetap_target = np.linspace(0.01, 84.0, n_x)  # “thetap = fliplr(linspace(0.01,84,1000))” in MATLAB (we use increasing)
    TEC_err_grid = np.zeros((len(e_vec), n_x), dtype=float)

    for i, e in enumerate(e_vec):
        f = ra*e
        rb = np.sqrt(max(ra*ra - f*f, 0.0))
        xe = np.linspace(-ra, f, n_x)
        # ye can become tiny; guard negatives from roundoff
        inner = 1.0 - (xe*xe)/(ra*ra)
        inner = np.where(inner < 0, 0, inner)
        ye = np.sqrt((rb*rb) * inner)

        # incidence angle vs x
        theta = np.degrees(np.arctan2(np.abs(xe - f), np.maximum(ye, 1e-30)))  # deg

        # footpoint height (|yf|)
        inner_f = (f*f)/(ra*ra) - 1.0
        yf = np.sqrt(np.abs((rb*rb) * inner_f))

        two_wayTEC = 2.0*np.abs(yf)
        ellipseTEC = np.abs(yf) + np.sqrt(ye*ye + (xe - f)*(xe - f))
        TEC_err = ellipseTEC - two_wayTEC  # per-x error

        # Interpolate onto a common theta grid (increasing); if theta not monotone, sort first
        order = np.argsort(theta)
        theta_sorted = theta[order]
        err_sorted = TEC_err[order]
        TEC_err_grid[i, :] = np.interp(thetap_target, theta_sorted, err_sorted, left=err_sorted[0], right=err_sorted[-1])

    return thetap_target, np.asarray(e_vec), TEC_err_grid

def plot_peters_fig7_like(angles_deg, e_vec, TEC_err, out_png):
    """
    “imagesc(thetap, e, TEC_error)” analog with axis xy and caxis limits like the MATLAB.
    """
    vmin, vmax = 0.0, 2.68e14
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.pcolormesh(angles_deg, e_vec, TEC_err, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Angle of Incidence (Degrees)")
    ax.set_ylabel("Ionosphere Eccentricity")
    ax.set_title("Maximum Incidence Angle for TEC Error (Peters toy model)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("TEC Error (m$^{-2}$)")
    ax.set_xlim(0, 90)
    ax.set_ylim(e_vec.min(), min(e_vec.max(), 0.625))  # match MATLAB's display
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig, ax


# ---------------------- δ(Δt) helper (optional) -----------------
def delta_dt_branch(delta_t_vhf, delta_t_hf, D_hf, lats, alts_m,
                    f_c_hf=F_C_HF, bw_hf=BW_HF,
                    n_iters=N_ITERS, relax=RELAX):
    # Match a single VHF reference to all HF rays (same as your current main)
    origin_lat = None  # first HF ray's starting latitude
    # For simplicity: assume first HF ray corresponds to SAT_LATS[0] origin
    # In your main, you matched by ground track; here we replicate that behavior:

    # Build a VHF array aligned with HF rays by taking the VHF Δt at the origin lat.
    # (Your current code: takes the first HF origin latitude and finds nearest VHF track.)
    # We'll mimic that exactly:

    # Use SAT_LATS[0] as origin; map to nearest VHF index
    sat_lats_vhf = SAT_LATS_VHF
    origin_lat = SAT_LATS_VHF[0]
    origin_lat_idx = int(np.argmin(np.abs(sat_lats_vhf - origin_lat)))
    delta_t_vhf_matched = np.full(len(delta_t_hf), delta_t_vhf[origin_lat_idx])

    # Convert δ(Δt) to TEC with the HF frequency pair
    c = 3e8
    f1_h = f_c_hf - 0.5*bw_hf
    f2_h = f_c_hf + 0.5*bw_hf
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)

    delta_dt_all = -(delta_t_vhf_matched - delta_t_hf)
    tec_est_ddt_all = (c * delta_dt_all / K_IONO) * freq_factor_h
    # tec_est_ddt_all = (c * delta_dt_all / 80.6) * freq_factor_h

    Ne_rec_ddt = reconstruct_art_sparse(D_hf, tec_est_ddt_all, len(lats), len(alts_m), n_iters, relax)
    return Ne_rec_ddt

def centers_to_edges(c):
    c = np.asarray(c)
    e = np.empty(c.size + 1, dtype=c.dtype)
    e[1:-1] = 0.5*(c[1:] + c[:-1])
    e[0]     = c[0] - 0.5*(c[1] - c[0])
    e[-1]    = c[-1] + 0.5*(c[-1] - c[-2])
    return e

# ---------------------- Function calls ----------------------------
def run_pipeline():
    # Step 0
    print("Step 0: Load Data")
    alt_km_1d, Ne_1d = step0_load_dataframe()

    # Step 1
    print("Step 1: Build Ionosphere")
    lats, alt_km, alts_m, iono = step1_build_iono(alt_km_1d, Ne_1d, lat_extent=(-10, 10), lat_res=200)

    # Step 2
    print("Step 2: Generate Rays")
    if not USE_GPU:
        rays_vhf, integ_vhf, rays_hf, theta_per_ray, integ_hf = step2_generate_rays(lats, alts_m)
    else:
        rays_hf, theta_per_ray, integ_hf = step2_generate_rays_gpu_sweep(
        lats, alts_m,
        theta_list_hf=THETA_LIST_HF,  # or your list
        sat_lats=SAT_LATS_HF,  # np.linspace(-6.0, 6.0, 181),   # denser tracks for richer TEC
        npts_hf=500,
        t_sweep= INTEG_TIMES_HF_PER_ANGLE  # np.linspace(0.05, 0.15, 21)     # dense T grid (x-axis)
        )
        rays_vhf = trace_passive_nadir(SAT_LATS_VHF, alts_m, npts=NPTS_VHF)
        integ_vhf = np.full(len(rays_vhf), INTEG_TIMES_VHF, dtype=float)

    # Step 3
    if not USE_GPU:
        vtec_vhf, delta_t_vhf, vtec_hf, delta_t_hf = step3_forward_model(
            iono, lats, alts_m, rays_vhf, integ_vhf, rays_hf, theta_per_ray, integ_hf
        )
    else:
        # D_hf = build_geometry_matrix_weighted_sparse(rays_hf, lats, alts_m, dtype=np.float32)
        # D_vhf = build_geometry_matrix_weighted_sparse(rays_vhf, lats, alts_m, dtype=np.float32)
        D_vhf, D_hf = step4_build_geometry(rays_vhf, rays_hf, lats, alts_m)
        print("HF geometry:", D_hf.shape, "nnz=", D_hf.nnz, "density=", D_hf.nnz/(D_hf.shape[0]*D_hf.shape[1]))
        print("VHF geometry:", D_vhf.shape, "nnz=", D_vhf.nnz, "density=", D_vhf.nnz/(D_vhf.shape[0]*D_vhf.shape[1]))
        D_all = scipy.sparse.vstack([D_vhf, D_hf], format="csr")
        # D_all_gpu = cpx_sparse.csr_matrix(D_all)

        tec_true, vtec_hf, delta_t_hf = step3_forward_model_gpu(
        iono=iono, lats=lats, alts_m=alts_m,
        D_hf=D_hf,
        theta_per_ray=theta_per_ray,
        integ_hf=integ_hf,
        f_c_hf=F_C_HF, bw_hf=BW_HF
        )
        # VHF path (CPU; cheap)
        stec_vhf = np.array([compute_STEC_along_path(iono, lats, alts_m, ray) for ray in rays_vhf])
        vtec_vhf, delta_t_vhf = dualfreq_to_VTEC(
            stec_slant=stec_vhf,
            f_c=F_C_VHF,
            bw=BW_VHF,
            theta_deg=0.0,
            integration_time=integ_vhf,
            return_deltat=True
        )

    if TEST_CHECKS:
        # === Step 3 math checks (mirrors unit_test) ===
        print("\n[UNIT] Running Step 3 math checks (inline)...")

        # (A) Eq.9 check at theta = 0: VTEC should equal slant TEC for a vertical path with huge T (no noise).
        TEC_test = np.array([1.23e16])
        VTEC_test, _ = dualfreq_to_VTEC(
            stec_slant=TEC_test, f_c=F_C_HF, bw=BW_HF,
            theta_deg=0.0, integration_time=1e9, return_deltat=True
        )
        assert np.allclose(VTEC_test[0], TEC_test[0], rtol=1e-6), "[Eq.9] VTEC != TEC (theta=0)"

        # (B) Verticalization: VTEC ≈ TEC_slant * cos(theta)
        th = 60.0
        VTEC_angle, _ = dualfreq_to_VTEC(
            stec_slant=TEC_test, f_c=F_C_HF, bw=BW_HF,
            theta_deg=th, integration_time=1e9, return_deltat=True
        )
        expected = TEC_test[0]*np.cos(np.deg2rad(th))
        np.testing.assert_allclose(VTEC_angle[0], expected, rtol=1e-6, atol=1e-2)

        # (C) δt(f) scaling: HF should give much larger |δt| than VHF for same TEC
        def _delta_t_mag(f0, bw):
            _, dt = dualfreq_to_VTEC(TEC_test, f_c=f0, bw=bw, theta_deg=0.0, integration_time=1e9, return_deltat=True)
            return float(np.abs(dt.mean()))
        dt_hf  = _delta_t_mag(F_C_HF,  BW_HF)
        dt_vhf = _delta_t_mag(F_C_VHF, BW_VHF)
        assert dt_hf > 5*dt_vhf, f"|δt| scaling unexpected: HF={dt_hf:.2e}, VHF={dt_vhf:.2e}"

        # (D) Noise scaling with T and BW (σ_dt ∝ 1/(BW * sqrt(T)))
        TEC_many = np.full(4000, TEC_test[0])
        def _std_dt(bw, T):
            _, dt = dualfreq_to_VTEC(TEC_many, f_c=F_C_HF, bw=bw, theta_deg=0.0, integration_time=T, return_deltat=True)
            return float(np.std(dt))
        s1 = _std_dt(BW_HF,     0.05)
        s2 = _std_dt(BW_HF,     0.20)  # 4x T → ≈ half σ
        s3 = _std_dt(BW_HF*2.0, 0.05)  # 2x BW → ≈ half σ
        assert s2 < 0.6*s1, f"T scaling off: σ(4T) ~ 0.5σ(T), observed {s2/s1:.2f}"
        assert s3 < 0.6*s1, f"BW scaling off: σ(2BW) ~ 0.5σ(BW), observed {s3/s1:.2f}"

        # (E) Consistency: D@x vs explicit STEC on a UNIFORM ionosphere (robust to discretization)
        Ne0 = 5e9
        iono_const = np.full_like(iono, Ne0, dtype=float)

        # GPU-style vector for D@x (match your D column order)
        x_const = iono_const.astype(np.float32).ravel(order="C")
        tec_true_const = (D_hf @ x_const).astype(np.float64)  # CPU multiply is fine here

        sample = slice(0, min(200, len(rays_hf)))  # bigger sample still fast
        stec_explicit_const = np.array(
            [compute_STEC_along_path(iono_const, lats, alts_m, r) for r in rays_hf[sample]]
        )

        rel_err = np.linalg.norm(tec_true_const[sample] - stec_explicit_const) / max(
            np.linalg.norm(tec_true_const[sample]), 1.0
        )
        assert rel_err < 1e-2, f"D@x vs explicit STEC mismatch on uniform iono: rel_err={rel_err:.3e}"

        print("[OK] Step 3 math checks passed (inline).")

        # The same DIAG prints (shapes & δt stats) you used in the unit test
        print("\n[DIAG] Step 3 checks")
        print(f"VHF: n_rays={len(rays_vhf)}, integ_vhf.shape={integ_vhf.shape}")
        print(f"HF : n_rays={len(rays_hf)}, theta_per_ray.shape={theta_per_ray.shape}, integ_hf.shape={integ_hf.shape}")
        print(f"vtec_vhf.shape={vtec_vhf.shape}, delta_t_vhf.shape={delta_t_vhf.shape}")
        print(f"vtec_hf.shape={vtec_hf.shape},   delta_t_hf.shape={delta_t_hf.shape}")
        print(f"VHF Δt stats: mean={np.mean(delta_t_vhf):.3e} s, std={np.std(delta_t_vhf):.3e}, "
            f"min={np.min(delta_t_vhf):.3e}, max={np.max(delta_t_vhf):.3e}")
        print(f"HF  Δt stats: mean={np.mean(delta_t_hf):.3e} s, std={np.std(delta_t_hf):.3e}, "
            f"min={np.min(delta_t_hf):.3e}, max={np.max(delta_t_hf):.3e}")

        # Verticalization RMSE (same computation as unit test)
        theta_rad = np.deg2rad(np.asarray(theta_per_ray, float))
        stec_est = (vtec_hf / np.cos(theta_rad))
        err = stec_est - tec_true
        rmse = float(np.sqrt(np.mean(err**2)))
        rel  = rmse / max(float(np.mean(tec_true)), 1.0)
        print(f"[Step3] HF verticalization: RMSE={rmse:.3e} ({rel:.2%} of mean STEC)")

    plot_dt_vs_invf2(delta_t_hf, delta_t_vhf, F_C_HF, F_C_VHF)
    plot_vtec_recovery_vs_angle(vtec_true=5e14, angles_deg=np.arange(0,85,5), f0=F_C_HF, bw=BW_HF, T=1.0, dualfreq_to_VTEC=dualfreq_to_VTEC)

    # Step 4
    print("Step 4: Build Geometry Matrices")
    if not USE_GPU:
        D_vhf, D_hf = step4_build_geometry(rays_vhf, rays_hf, lats, alts_m)
        print("HF geometry:", D_hf.shape, "nnz=", D_hf.nnz, "density=", D_hf.nnz/(D_hf.shape[0]*D_hf.shape[1]))
        print("VHF geometry:", D_vhf.shape, "nnz=", D_vhf.nnz, "density=", D_vhf.nnz/(D_vhf.shape[0]*D_vhf.shape[1]))
        
    if TEST_CHECKS:
        # Step 4 diagnostics mirroring unit test
        print("[Step4] grid: n_lat={}, n_alt={}, n_vox={}".format(len(lats), len(alts_m), D_hf.shape[1]))
        hf_nz_per_row = np.diff(D_hf.indptr) if sp.issparse(D_hf) else np.count_nonzero(D_hf, axis=1)
        vhf_nz_per_row = np.diff(D_vhf.indptr) if sp.issparse(D_vhf) else np.count_nonzero(D_vhf, axis=1)
        print("[Step4] HF per-ray intersections: mean={:.1f}, std={:.1f}, min={}, max={}".format(
            np.mean(hf_nz_per_row), np.std(hf_nz_per_row), int(np.min(hf_nz_per_row)), int(np.max(hf_nz_per_row))))
        print("[Step4] VHF per-ray intersections: mean={:.1f}, std={:.1f}, min={}, max={}".format(
            np.mean(vhf_nz_per_row), np.std(vhf_nz_per_row), int(np.min(vhf_nz_per_row)), int(np.max(vhf_nz_per_row))))

        hf_touched = int((D_hf.sum(axis=0) if not sp.issparse(D_hf) else np.array(D_hf.getnnz(axis=0))).astype(bool).sum())
        vhf_touched = int((D_vhf.sum(axis=0) if not sp.issparse(D_vhf) else np.array(D_vhf.getnnz(axis=0))).astype(bool).sum())
        n_vox = D_hf.shape[1]
        print(f"[Step4] HF voxel coverage: touched={hf_touched}/{n_vox} ({100*hf_touched/n_vox:.1f}%)")
        print(f"[Step4] VHF voxel coverage: touched={vhf_touched}/{n_vox} ({100*vhf_touched/n_vox:.1f}%)")
        print(f"[Step4] densities: HF={D_hf.nnz/(D_hf.shape[0]*D_hf.shape[1]):.6f}, "
            f"VHF={D_vhf.nnz/(D_vhf.shape[0]*D_vhf.shape[1]):.6f}")

    plot_ray_coverage_heatmap(D_hf, lats, alts_m, "HF ray coverage")
    plot_ray_coverage_heatmap(D_vhf, lats, alts_m, "VHF ray coverage")

    # Step 5A (ARTs) + collect for weighted
    print("Step 5: Reconstruct")
    Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, D_all, vtec_all = step5_reconstruct(
        D_vhf, D_hf, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf, lats, alts_m,
        n_iters=N_ITERS, relax=RELAX
    )

    vtec_all = np.concatenate([vtec_vhf, vtec_hf]).astype(float)

    Ne_rec_all_hist, rmse_hist_unw, reg_hist_unw = art_with_history(
        D_all, vtec_all,
        n_lat=len(lats), n_alt=len(alts_m),
        n_iters=N_ITERS, relax=RELAX,
        compute_lcurve=True, L=None   # set L later if you add a regularizer
    )

    # Ne_rec_all_hist, rmse_hist_unw, reg_hist_unw = sirt_cimmino_with_history(
    #     D_all, vtec_all, n_lat=len(lats), n_alt=len(alts_m),
    #     n_iters=N_ITERS, relax=RELAX
    # )

    # Step 5B (Weighted ART)
    Ne_rec_weighted, weights = step5_weighted_reconstruct(
        D_all, vtec_all, integ_vhf, integ_hf, bw_vhf=BW_VHF, bw_hf=BW_HF,
        n_lat=len(lats), n_alt=len(alts_m), n_iters=50, relax=0.2
    )

    Ne_rec_weighted_hist, rmse_hist_w, reg_hist_w = weighted_art_with_history(
        D_all, vtec_all, weights,
        n_lat=len(lats), n_alt=len(alts_m),
        n_iters=N_ITERS, relax=RELAX,
        compute_lcurve=True, L=None
    )
    plot_rmse_vs_iteration(rmse_hist_unw, rmse_hist_w)
    if reg_hist_unw is not None and reg_hist_w is not None:
        # You can plot both on the same axes by calling twice, or make two panels
        plot_lcurve(norm_residual=rmse_hist_unw, norm_regularizer=reg_hist_unw, title="L-curve (unweighted)")
        plot_lcurve(norm_residual=rmse_hist_w,  norm_regularizer=reg_hist_w,  title="L-curve (weighted)")

    # Optional δ(Δt) branch (same behavior as your script)
    Ne_rec_ddt = delta_dt_branch(delta_t_vhf, delta_t_hf, D_hf, lats, alts_m,
                                 f_c_hf=F_C_HF, bw_hf=BW_HF,
                                 n_iters=N_ITERS, relax=RELAX)
                                 
    fig, ax, r_all, r_w = plot_residual_histograms(D_all, vtec_all, Ne_rec_all, Ne_rec_weighted)
    plot_residuals_vs_angle_time(D_all, vtec_all, D_vhf, theta_per_ray, integ_hf, Ne_rec_all)

    # Ensure arrays are (n_alt, n_lat) for cross-sections (transpose if yours are (n_lat,n_alt))
    truth_alt_lat = iono
    rec_all_alt_lat = Ne_rec_all
    rec_w_alt_lat   = Ne_rec_weighted
    plot_recon_cross_sections(lats, alts_m, iono.T, Ne_rec_all, Ne_rec_weighted)
    plot_voxel_uncertainty_proxy(D_all, lats, alts_m)

    # Step 6 (plots)
    print("Step 6: Plotting & Evaluation")
    plot_recons(
        lats, alts_m,
        [Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, Ne_rec_weighted, Ne_rec_ddt],
        ["VHF only", "HF only", "Combined (standard)", "Combined (weighted)", "δ(Δt)-based"]
    )
    plot_raypaths(
    lats=lats,
    alts_m=alts_m,
    rays_hf=rays_hf,
    theta_per_ray=theta_per_ray,
    rays_vhf=rays_vhf,
    # optional: limit lines per angle to reduce clutter
    # max_rays_per_angle=60,
    )

    if not USE_GPU:
        t_meas, tec_for_bin, delta_tec_err_passive, range_error_after_passive = build_passive_plot_inputs(
            iono=iono, lats=lats, alts_m=alts_m,
            rays_hf=rays_hf,
            integ_hf=integ_hf,
            theta_per_ray=theta_per_ray,
            vtec_hf=vtec_hf,
            delta_t_hf=delta_t_hf,
            delta_t_vhf=delta_t_vhf,
        )
        # Then make the two figures
        plot_passive_tec_error_heatmap_from_arrays(t_meas, tec_for_bin, delta_tec_err_passive, smooth_sigma=1.2)
        plot_vhf_range_error_heatmap_from_arrays(t_meas, tec_for_bin, range_error_after_passive, smooth_sigma=1.2)

    else:
        t_meas, tec_for_bin, delta_tec_err_passive, range_error_after_passive = build_passive_plot_inputs_GPU(
            iono=iono, lats=lats, alts_m=alts_m,
            D_hf=D_hf,                      # use GPU @ for tec_true
            integ_hf=integ_hf,
            theta_per_ray=theta_per_ray,
            vtec_hf=vtec_hf,
            delta_t_hf=delta_t_hf,
            delta_t_vhf=delta_t_vhf,
            sat_lats_vhf=SAT_LATS_VHF,
            hf_origin_lat=float(SAT_LATS_HF[0]),
        )
        plot_passive_tec_error_heatmap_GPU(t_meas, tec_for_bin, delta_tec_err_passive, smooth_sigma=1.2)
        plot_vhf_range_error_heatmap_GPU(t_meas, tec_for_bin, range_error_after_passive, smooth_sigma=1.2)

    # --- Peters Fig. 7 validation surface ---
    angles_deg, ecc_vec, TEC_err = peters_theta_tecerror_surface(
        e_vec=np.linspace(0.01, 0.99, 200), ra=4e15, n_x=1000
    )
    plot_peters_fig7_like(angles_deg, ecc_vec, TEC_err, os.path.join(IMG_DIR, "Z_peters_fig7_surface.png"))

    # (optional) reproduce the small dTEC / Δt demo at the top of the MATLAB
    dTEC = peters_dtec_from_range(dr=3.0, fv=F_C_VHF, fh=F_C_HF)  # same constants you use
    TEC_true = 4e15
    dt_true  = peters_dt_from_TEC(TEC_true, fv=F_C_VHF, fh=F_C_HF)
    dt_error = peters_dt_from_TEC(TEC_true + dTEC, fv=F_C_VHF, fh=F_C_HF)
    dt_dTEC  = dt_error - dt_true
    VTEC_dTEC = (dt_dTEC*2*np.pi) * ((F_C_HF**2)*(F_C_VHF**2)) / (1.69e-6*((F_C_VHF**2)-(F_C_HF**2)))
    print(f"[Peters demo] dTEC={dTEC:.3e}, dt_true={dt_true:.3e}s, dt_error={dt_error:.3e}s, dt_dTEC={dt_dTEC:.3e}s, VTEC_dTEC={VTEC_dTEC:.3e}")


    return {
        "lats": lats,
        "alts_m": alts_m,
        "iono": iono,
        "vtec_vhf": vtec_vhf,
        "vtec_hf": vtec_hf,
        "delta_t_vhf": delta_t_vhf,
        "delta_t_hf": delta_t_hf,
        "theta_per_ray": theta_per_ray,
        "integ_vhf": integ_vhf,
        "integ_hf": integ_hf,
        "D_vhf": D_vhf,
        "D_hf": D_hf,
        "D_all": D_all,
        "Ne_rec_vhf": Ne_rec_vhf,
        "Ne_rec_hf": Ne_rec_hf,
        "Ne_rec_all": Ne_rec_all,
        "Ne_rec_weighted": Ne_rec_weighted,
        "Ne_rec_ddt": Ne_rec_ddt,
        "weights": weights,
        "vtec_all": vtec_all,
    }

# ---------------------- Entry point -----------------------------
if __name__ == "__main__":
    results = run_pipeline()
    # You can add any post-analysis you already do here (heatmaps, θ studies, etc.)