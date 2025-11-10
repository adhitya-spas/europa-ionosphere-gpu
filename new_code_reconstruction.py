# ---------------------- Imports ----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy
import scipy.sparse as sp 
import os
from datetime import datetime
from physics_constants import C_LIGHT, C_IONO, R_EUROPA
import gc

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
from plot_scripts import plot_raypaths, plot_ionosphere_and_enhancement, plot_recons, _print_ne_stats

USE_GPU = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import scipy.sparse as sp
except Exception:
    USE_GPU = False
    cp = None
    cpx_sparse = None

# Keep only minimal constants and thinning helper at top; full GPU/sparse helpers
# are defined later in the file so they can rely on local imports and flags.
MAX_ROWS_CHUNKED = 12000   # tune for your box

MAX_HF_SHOTS = 8000  # tune; keeps CSR reasonable

def thin_by_stride(items, max_keep):
    if len(items) <= max_keep:
        return items
    stride = int(np.ceil(len(items) / max_keep))
    return items[::stride]
# ---------------------- Constants ----------------------

# VHF & HF radio parameters
F_C_VHF = 60e6        # VHF center frequency [Hz]
BW_VHF  = 10e6        # VHF bandwidth [Hz]
F_C_HF  = 9e6         # HF center frequency [Hz]
BW_HF   = 1e6         # HF bandwidth [Hz]

# Ionosphere grid parameters
LAT_EXTENT = (-10.0, 10.0)   # degrees
LAT_RES    = 200              # number of latitude bins

# Spacecraft parameters
SC_VELOCITY = 2000.0          # m/s
SC_ALTITUDE = 1500.0*1e3      # m

# Reconstruction parameters
# ART settings
N_ITERS = 20
RELAX   = 0.1

# ---------------------- DEFINES ----------------------

# Ray tracing mode
RAY_TRACE_MODE = "left_fixed"  # "sat_right", "left_fixed","mirror"

# Display flags
STEP_1_PLOT_IONO = True
STEP_2_PLOT_RAYS = True
STEP_6_PLOT_RECONS = True

# ---------------------- Step 1: Load Ionosphere -----------------------
def load_dataframe_and_build_ionosphere(s_no, enhancement_amplitude,path="new_mission_df.pkl", mission_name="E6a Exit"):
    # Loading data
    df = load_mission_df(path)
    row = df[df["Mission"] == mission_name].iloc[0]
    alt_km_1d, Ne_1d = row["Altitude"], row["Ne"] #km and cm^-3

    # Building ionosphere
    lats, alt_km, iono = build_ionosphere(
        pd.DataFrame({"altitude": alt_km_1d, "ne": Ne_1d*1e6}),  # convert cm^-3 to m^-3
        lat_extent=LAT_EXTENT, lat_res=LAT_RES)

    # Add Gaussian enhancement
    enhanced_iono = add_gaussian_enhancement(
            ionosphere_map=iono,
            latitudes=lats,
            altitude=alt_km,
            lat_center=0.0,
            alt_center=150.0,
            lat_width=1.0,
            alt_width=20.0,
            amplitude=enhancement_amplitude
        )
    
    # Plot ionosphere and enhancement
    if STEP_1_PLOT_IONO:
        plot_ionosphere_and_enhancement(s_no, lats, alt_km, iono, enhanced_iono)
    
    return lats, alt_km, enhanced_iono

# ---------------------- Step 2: Generate Ray Paths (GPU) -----------------------
def generate_ray_paths(s_no, sat_lats_vhf, alts_m, npts_vhf,
                       sat_lats_hf, incidence_angle_deg, npts_hf,
                       integ_vhf, integ_hf, iono, lats):

    # VHF rays (nadir)
    rays_vhf = trace_passive_nadir(sat_lats_vhf, alts_m, npts=npts_vhf)

    # HF rays (oblique) — build a flat list of rays and matching theta per ray
    rays_hf = []
    theta_per_ray = []
    for theta in np.atleast_1d(incidence_angle_deg):
        rays_hf_theta = trace_passive_oblique(
            sat_lats=sat_lats_hf,
            h0_m=alts_m.max(),
            hsat_m=alts_m.max(),
            theta_i_deg=theta,
            npts=npts_hf,
            R_E_m=R_EUROPA,
            mode=RAY_TRACE_MODE,
            lat_left_deg=0.0          # << choose your left start latitude
        )
        # rays_hf_theta is a list of rays (one per sat_lats_hf entry)
        for ray in rays_hf_theta:
            rays_hf.append(ray)
            theta_per_ray.append(float(theta))

    # Build per-ray integration times (support scalar or array of T values)
    Tlist = np.atleast_1d(integ_hf)
    if Tlist.size == 1:
        integ_hf_per_ray = np.full(len(rays_hf), float(Tlist[0]), dtype=float)
    else:
        # replicate entire ray set for each T in Tlist
        rays_rep = []
        theta_rep = []
        integ_rep = []
        for T in Tlist:
            rays_rep.extend(rays_hf)
            theta_rep.extend(theta_per_ray)
            integ_rep.extend([float(T)] * len(rays_hf))
        rays_hf = rays_rep
        theta_per_ray = theta_rep
        integ_hf_per_ray = np.array(integ_rep, dtype=float)

    # Thin HF rays to cap CSR size (prevents explosion for small theta)
    rays_hf = thin_by_stride(rays_hf, MAX_HF_SHOTS)
    theta_per_ray = np.asarray(theta_per_ray, float)[: len(rays_hf)]
    integ_hf_per_ray = np.asarray(integ_hf_per_ray, float)[: len(rays_hf)]

    # Build Geometry Matrices
    # Use the latitude grid `lats` (not satellite lat arrays) so columns match ionosphere pixels
    D_vhf = build_geometry_matrix_weighted_sparse(rays_vhf, lats, alts_m, dtype=np.float32)
    D_hf  = build_geometry_matrix_weighted_sparse(rays_hf,  lats, alts_m, dtype=np.float32)
    D_all = scipy.sparse.vstack([D_vhf, D_hf], format="csr")

    # HF path (chunked multiply to reduce GPU/CPU peak memory)
    # Build a contiguous float32 true-ionosphere vector
    x_true_vector = np.asarray(iono, dtype=np.float32).ravel(order="C")
    tec_true = chunked_sparse_mv(D_hf, x_true_vector, rows_per_chunk=4000, use_gpu=USE_GPU, dtype=np.float32)
    # immediately free to keep peak memory low
    free_all_gpu(); gc.collect()
    # Integration times and angles per ray
    vtec_hf, delta_t_hf = dualfreq_to_VTEC(
        stec_slant=tec_true,
        f_c=F_C_HF,
        bw=BW_HF,
        theta_deg=np.asarray(theta_per_ray, float),
        integration_time=np.asarray(integ_hf_per_ray, float),
        return_deltat=True
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

    # Print Set of Rays HF and VHF || Plot the Rays
    if STEP_2_PLOT_RAYS:
        # Just plot the rays
        plot_raypaths(
            s_no=s_no,
            lats=lats,
            alts_m=alts_m,
            rays_hf=rays_hf,
            theta_per_ray=theta_per_ray,
            rays_vhf=rays_vhf,
            #max_rays_per_angle=5  # limit for clarity
        )
        # Print number of rays and geometry info
        print(f"Generated {len(rays_vhf)} VHF rays and {len(rays_hf)} HF rays.")
        # Geometry matrix info
        print("HF geometry:", D_hf.shape, "nnz=", D_hf.nnz, "density=", D_hf.nnz/(D_hf.shape[0]*D_hf.shape[1]))
        print("VHF geometry:", D_vhf.shape, "nnz=", D_vhf.nnz, "density=", D_vhf.nnz/(D_vhf.shape[0]*D_vhf.shape[1]))
        # STEC and VTEC check
        print("HF STEC (first 5):", tec_true[:5])
        print("HF VTEC (first 5):", vtec_hf[:5])
        print("VHF STEC (first 5):", stec_vhf[:5])
        print("VHF VTEC (first 5):", vtec_vhf[:5])

    # NOTE: return per-ray θ and T so downstream stays aligned with D_hf
    return rays_vhf, rays_hf, D_all, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf, D_vhf, D_hf, np.asarray(theta_per_ray, float), np.asarray(integ_hf_per_ray, float)

# ---------------------- Step 3: Forward Model (GPU) -----------------------
def forward_model_gpu(
    iono, lats, alts_m,
    D_hf,                 # geometry matrix for HF rays (SciPy CSR or dense np.ndarray)
    theta_per_ray,        # (N,) deg from vertical
    integ_hf,             # (N,) seconds
    f_c_hf=F_C_HF, bw_hf=BW_HF,
):
    if not USE_GPU:
        raise RuntimeError("CuPy not available; use the CPU path or set USE_GPU=True.")
    # Use the chunked sparse multiply to avoid building a full GPU matrix at once
    x_true_vector = np.asarray(iono, dtype=np.float32).ravel(order="C")
    tec_true = chunked_sparse_mv(D_hf, x_true_vector, rows_per_chunk=4000, use_gpu=True, dtype=np.float32)
    free_all_gpu(); gc.collect()

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

# ---------------------- Step 4: BUILD GEOMETRY MATRICES -----------------------
def step5_reconstruct(D_vhf, D_hf, vtec_vhf, vtec_hf,
                      delta_t_vhf, delta_t_hf,
                      lats, alts_m,
                      n_iters=N_ITERS, relax=RELAX):
    
    n_lat, n_alt = len(lats), len(alts_m)
    # 5A) ART (VHF-only, HF-only, stacked)
    # VHF-only
    if sp.isspmatrix_csr(D_vhf) and D_vhf.shape[0] > MAX_ROWS_CHUNKED:
        Ne_rec_vhf, _ = art_kaczmarz_chunked(
            D_vhf, vtec_vhf, n_lat=n_lat, n_alt=n_alt,
            n_iters=n_iters, relax=relax,
            rows_per_chunk=4000, nonneg=True, dtype=np.float32
        )
    else:
        if sp.issparse(D_vhf):
            Ne_rec_vhf = reconstruct_art_sparse(D_vhf, vtec_vhf, n_lat, n_alt, n_iters, relax)
        else:
            Ne_rec_vhf = reconstruct_art(D_vhf, vtec_vhf, n_lat, n_alt, n_iters, relax)

    # HF-only
    if sp.isspmatrix_csr(D_hf) and D_hf.shape[0] > MAX_ROWS_CHUNKED:
        Ne_rec_hf, _ = art_kaczmarz_chunked(
            D_hf, vtec_hf, n_lat=n_lat, n_alt=n_alt,
            n_iters=n_iters, relax=relax,
            rows_per_chunk=4000, nonneg=True, dtype=np.float32
        )
    else:
        if sp.issparse(D_hf):
            Ne_rec_hf = reconstruct_art_sparse(D_hf, vtec_hf, n_lat, n_alt, n_iters, relax)
        else:
            Ne_rec_hf = reconstruct_art(D_hf, vtec_hf, n_lat, n_alt, n_iters, relax)

    # Combined
    D_all = sp.vstack([D_vhf, D_hf], format="csr") if (sp.issparse(D_vhf) or sp.issparse(D_hf)) \
            else np.vstack([D_vhf, D_hf])
    vtec_all = np.concatenate([vtec_vhf, vtec_hf])

    if sp.isspmatrix_csr(D_all) and D_all.shape[0] > MAX_ROWS_CHUNKED:
        Ne_rec_all, _ = art_kaczmarz_chunked(
            D_all, vtec_all, n_lat=n_lat, n_alt=n_alt,
            n_iters=n_iters, relax=relax,
            rows_per_chunk=4000, nonneg=True, dtype=np.float32
        )
    else:
        if sp.issparse(D_all):
            Ne_rec_all = reconstruct_art_sparse(D_all, vtec_all, n_lat, n_alt, n_iters, relax)
        else:
            Ne_rec_all = reconstruct_art(D_all, vtec_all, n_lat, n_alt, n_iters, relax)
    return Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, D_all, vtec_all

def step4B_weighted_reconstruct(D_all, vtec_all, integ_vhf, integ_hf, bw_vhf=BW_VHF, bw_hf=BW_HF,
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

def _rmse_from(D, x_c, y):
    """RMSE of residuals r = D x - y."""
    if sp.issparse(D):
        r = D @ x_c
    else:
        r = D.dot(x_c)
    r -= y
    return float(np.sqrt(np.mean((r)**2)))

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
    if sp.issparse(D) and not sp.isspmatrix_csr(D):
        D = D.tocsr()
    if sp.issparse(D):
        row_norm2 = np.array(D.power(2).sum(axis=1)).ravel()
    else:
        row_norm2 = np.sum(D * D, axis=1)
    row_norm2[row_norm2 == 0.0] = 1.0
    rmse_hist, reg_hist = [], []
    for it in range(n_iters):
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

# ---------------------- Step 0: Correlation between theta and integration time -----------------------

def plan_transect_fixed_length(theta_grid_deg, total_km,
                               lat_start_deg, R_E_km, h_km, v_km_s): # Irregular sweeping
    """
    MODE A: Build a transect with *equal total length* for a given theta grid.
    We step through theta_grid in order (repeating if needed), compute Ti from
    Ti = (2h/v)*tan(theta), convert to latitude step, and accumulate until the
    sum of v*Ti reaches 'total_km'.

    Returns:
      sat_lats   : np.ndarray [N+1], lat_start .. lat_end (deg)
      theta_list : np.ndarray [N],   θ per shot (deg)
      T_list     : np.ndarray [N],   integration time per shot (s)
    """
    r_s = R_E_km + h_km
    K   = (2.0 * h_km) / v_km_s
    sat_lats = [float(lat_start_deg)]
    theta_list = []
    T_list = []
    dist_accum = 0.0
    i = 0
    while dist_accum < total_km:
        theta = float(theta_grid_deg[i % len(theta_grid_deg)])
        Ti = K * np.tan(np.radians(theta))
        di = v_km_s * Ti
        dphi_deg = (di / r_s) * (180.0 / np.pi)
        if dist_accum + di > total_km:
            remain = total_km - dist_accum
            frac = remain / di if di > 0 else 0.0
            sat_lats.append(sat_lats[-1] + frac * dphi_deg)
            theta_list.append(theta)
            T_list.append(frac * Ti)
            break
        sat_lats.append(sat_lats[-1] + dphi_deg)
        theta_list.append(theta)
        T_list.append(Ti)
        dist_accum += di
        i += 1
    return np.asarray(sat_lats), np.asarray(theta_list), np.asarray(T_list)

def scale_angles_to_match_length(theta_grid_deg, total_km, h_km, v_km_s): # Regular consecutive sweeps
    """
    MODE B: Same total length *and* same number of shots by scaling angles.
    We find γ so that sum( Ti' ) = total_km / v, with Ti' = (2h/v)*tan(θ').
    Choose tan(θ') = γ * tan(θ). Then:
        γ = (total_km / v) / ( (2h/v) * sum(tanθ) ) = total_km / (2h * sum(tanθ))
    Returns new_theta_deg (same length as input), and Ti' for those angles.
    """
    tan_list = np.tan(np.radians(theta_grid_deg))
    S = float(np.sum(tan_list))
    if S <= 0:
        raise ValueError("Sum of tan(theta) must be positive.")
    gamma = total_km / (2.0 * h_km * S)
    tan_new = gamma * tan_list
    theta_new = np.degrees(np.arctan(tan_new))
    K = (2.0 * h_km) / v_km_s
    T_new = K * tan_new
    return theta_new, T_new

def ensure_1d(x):
    """Return x as a 1D numpy array (None -> empty array)."""
    if x is None:
        return np.array([], dtype=float)
    arr = np.atleast_1d(x).astype(float)
    return arr

def sat_lats_from_times(T_list, lat_start_deg, R_E_km, h_km, v_km_s):
    r_s = R_E_km + h_km
    d_km = v_km_s * np.asarray(T_list)
    dphi = (d_km / r_s) * (180.0/np.pi)
    lats = [lat_start_deg]
    for step in dphi:
        lats.append(lats[-1] + step)
    return np.asarray(lats)

def plan_transect_single_angle(theta_deg, total_km, lat_start_deg,
                               R_E_km, h_km, v_km_s,
                               min_shots=2, clamp_last=True):
    """
    Build a transect using ONE fixed HF angle (theta_deg) for all shots.
    - Uses geometric Ti = (2h/v) * tan(theta) for each shot.
    - Repeats that shot until we reach (or nearly reach) total_km.
    - Optionally 'clamp_last': adjust only the *last* shot's time so that the
      total distance matches total_km exactly (keeps theta fixed).
    Returns: sat_lats (N+1), theta_per_ray (N), integ_per_ray (N)
    """
    theta_deg = float(theta_deg)
    tan_th = np.tan(np.radians(theta_deg))
    if tan_th <= 0:
        raise ValueError("theta must be > 0° for a downward-looking reflection.")
    K = (2.0 * h_km) / v_km_s                 # km / (1/tan)
    Ti = K * tan_th                           # seconds
    step_km = v_km_s * Ti                     # km per shot

    if step_km <= 0:
        raise ValueError("Computed step_km <= 0; check inputs.")

    # Number of full shots we can place before exceeding total_km
    N_full = int(max(min_shots, np.floor(total_km / step_km)))
    # Distance covered by those
    dist_full = N_full * step_km
    remaining_km = max(0.0, total_km - dist_full)

    T_list = [Ti] * N_full
    thetas = [theta_deg] * N_full

    if remaining_km > 1e-9:
        if clamp_last:
            # Add one final partial shot with shorter time so we exactly hit total_km
            T_last = remaining_km / v_km_s
            T_list.append(T_last)
            thetas.append(theta_deg)
        else:
            # Leave a small shortfall (no extra partial shot)
            pass

    T_arr = np.asarray(T_list, dtype=float)
    theta_arr = np.asarray(thetas, dtype=float)

    sat_lats = sat_lats_from_times(T_arr, lat_start_deg, R_E_km, h_km, v_km_s)
    return sat_lats, theta_arr, T_arr

def delta_dt_branch(delta_t_vhf, delta_t_hf, D_hf, lats, alts_m,
                    f_c_hf=F_C_HF, bw_hf=BW_HF,
                    n_iters=N_ITERS, relax=RELAX):
    # Combine delta_t from VHF+HF
    delta_all = np.concatenate([delta_t_vhf, delta_t_hf])
    # Convert Δt back to equivalent slant TEC (Eq. 9 inverted)
    f1, f2 = f_c_hf - bw_hf/2, f_c_hf + bw_hf/2
    freq_factor = (f1**2 * f2**2) / (f2**2 - f1**2)
    tec_slant = (C_LIGHT * delta_all / C_IONO) * freq_factor
    # Reconstruct Ne using the same ART kernel
    if sp.issparse(D_hf):
        Ne_rec_dt = reconstruct_art_sparse(D_hf, tec_slant,
                                           len(lats), len(alts_m),
                                           n_iters, relax)
    else:
        Ne_rec_dt = reconstruct_art(D_hf, tec_slant,
                                    len(lats), len(alts_m),
                                    n_iters, relax)
    return Ne_rec_dt

# ---------------------- HELPERS -----------------------

# ======== GPU + Sparse Utilities (place after imports) ========

import gc
import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

def free_all_gpu():
    """Release CuPy memory pools (safe to call even if CuPy unavailable)."""
    if _HAS_CUPY:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

def chunked_sparse_mv(D_csr, x, rows_per_chunk=4000, use_gpu=True, dtype=np.float32):
    """
    y = D @ x computed in row-chunks. Works with CPU or GPU.
    - D_csr: scipy.sparse.csr_matrix (float32 recommended)
    - x: 1D numpy array
    - rows_per_chunk: trade memory vs. speed
    - use_gpu: if True and CuPy is available, use GPU per chunk
    Returns numpy 1D array y.
    """
    assert sp.isspmatrix_csr(D_csr)
    x = np.asarray(x, dtype=dtype)
    m, n = D_csr.shape
    y_out = np.empty(m, dtype=dtype)

    if use_gpu and not _HAS_CUPY:
        use_gpu = False

    if use_gpu:
        x_gpu = cp.asarray(x)
    else:
        x_gpu = None

    for i0 in range(0, m, rows_per_chunk):
        i1 = min(i0 + rows_per_chunk, m)
        D_blk = D_csr[i0:i1].astype(dtype, copy=False)

        if use_gpu:
            Dg = cpx_sp.csr_matrix(D_blk)
            yg = Dg @ x_gpu
            y_out[i0:i1] = cp.asnumpy(yg)
            del Dg, yg
            free_all_gpu()
        else:
            y_out[i0:i1] = D_blk.dot(x)

        # Drop CPU chunk ASAP
        del D_blk
        gc.collect()

    if x_gpu is not None:
        del x_gpu
        free_all_gpu()

    return y_out

def csr_row_norm2(D):
    """Row-wise ||row||^2 for CSR or dense."""
    if sp.issparse(D):
        return np.array(D.power(2).sum(axis=1)).ravel().astype(np.float32)
    else:
        return np.sum(D * D, axis=1).astype(np.float32)

def art_kaczmarz_chunked(D_csr, y, *, n_lat, n_alt, n_iters=20, relax=0.1,
                         rows_per_chunk=4000, nonneg=True, dtype=np.float32):
    """
    Matrix-chunked Kaczmarz ART: iterate row-by-row but only keep a small
    CSR block in memory. Returns (x_map, rmse_hist).
    - D_csr: scipy.sparse.csr_matrix
    - y: numpy 1D
    """
    assert sp.isspmatrix_csr(D_csr)
    y = np.asarray(y, dtype=dtype)
    m, n = D_csr.shape
    x = np.zeros(n, dtype=dtype)

    for it in range(n_iters):
        row_start = 0
        while row_start < m:
            row_end = min(row_start + rows_per_chunk, m)
            Db = D_csr[row_start:row_end].astype(dtype, copy=False)
            yb = y[row_start:row_end]
            rn2 = csr_row_norm2(Db)
            rn2[rn2 == 0.0] = 1.0

            # Kaczmarz over rows in this block
            indptr = Db.indptr
            idxs   = Db.indices
            vals   = Db.data
            for i in range(row_end - row_start):
                s = indptr[i]; e = indptr[i+1]
                col = idxs[s:e]
                dat = vals[s:e]
                pred = float(np.dot(dat, x[col]))
                ri = yb[i] - pred
                alpha = relax * (ri / rn2[i])
                x[col] += alpha * dat

            del Db
            gc.collect()
            row_start = row_end

        if nonneg:
            np.maximum(x, 0, out=x)

    # no big D present anymore, compute RMSE one final time in chunks:
    rmse = _rmse_chunked(D_csr, x, y, rows_per_chunk=rows_per_chunk, dtype=dtype)
    return x.reshape(n_alt, n_lat, order="C"), [rmse]

def _rmse_chunked(D_csr, x, y, rows_per_chunk=4000, dtype=np.float32):
    """RMSE(Dx, y) computed via chunked matvec (CPU)."""
    yhat = chunked_sparse_mv(D_csr, x, rows_per_chunk=rows_per_chunk,
                             use_gpu=False, dtype=dtype)
    diff = (yhat - y).astype(dtype)
    return float(np.sqrt(np.mean(diff * diff)))



# ---------------------- MAIN PIPELINE ----------------------
def run_pipeline(s_no,
                 enhancement_amplitude,
                 sat_lats_vhf, npts_vhf,
                 sat_lats_hf, incidence_angle_deg, npts_hf,
                 integ_vhf, integ_hf):
    print("STEP 1")
    # STEP 1: Load ionosphere and build ionosphere map
    lats, alt_km, iono = load_dataframe_and_build_ionosphere(s_no, enhancement_amplitude)
    alts_m=alt_km*1e3

    print("STEP 2")
    # STEP 2: Generate ray paths (GPU)
    rays_vhf, rays_hf, D_all, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf, D_vhf, D_hf, theta_per_ray, integ_hf_per_ray = generate_ray_paths(
        s_no=s_no,
        sat_lats_vhf=sat_lats_vhf,
        alts_m=alts_m,
        npts_vhf=npts_vhf,
        sat_lats_hf=sat_lats_hf,
        incidence_angle_deg=incidence_angle_deg,
        npts_hf=npts_hf,
        integ_vhf=integ_vhf,
        integ_hf=integ_hf,
        iono=iono,
        lats=lats
    )
    print("Geometry matrices shapes:", D_vhf.shape, D_hf.shape)
    print("theta_per_ray length:", len(theta_per_ray))
    print("integ_hf_per_ray length:", len(integ_hf_per_ray))
    assert D_hf.shape[0] == len(theta_per_ray) == len(integ_hf_per_ray)

    print("STEP 3")
    # STEP 3: Forward model (GPU)
    tec_true_hf, vtec_hf_gpu, delta_t_hf_gpu = forward_model_gpu(
        iono=iono,
        lats=lats,
        alts_m=alts_m,
        D_hf=D_hf,
        theta_per_ray=theta_per_ray,
        integ_hf=integ_hf_per_ray
    )
    # keep memory low after big forward model
    gc.collect(); free_all_gpu()

    print("STEP 4")
    # STEP 4: RECONSTRUCTION (STEP 5 & 4B)
    Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, D_all, vtec_all = step5_reconstruct(
        D_vhf, D_hf, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf, lats, alts_m,
        n_iters=N_ITERS, relax=RELAX
    )
    print("STEP 4A")
    vtec_all = np.concatenate([vtec_vhf, vtec_hf]).astype(float)
    if sp.isspmatrix_csr(D_all) and D_all.shape[0] > MAX_ROWS_CHUNKED:
        Ne_rec_all_hist, rmse_hist_unw = art_kaczmarz_chunked(
            D_all, vtec_all, n_lat=len(lats), n_alt=len(alts_m),
            n_iters=N_ITERS, relax=RELAX,
            rows_per_chunk=4000, nonneg=True, dtype=np.float32
        )
        reg_hist_unw = None
    else:
        Ne_rec_all_hist, rmse_hist_unw, reg_hist_unw = art_with_history(
            D_all, vtec_all,
            n_lat=len(lats), n_alt=len(alts_m),
            n_iters=N_ITERS, relax=RELAX,
            compute_lcurve=True, L=None
        )
    gc.collect(); free_all_gpu()
    # Step 4B (Weighted ART)
    print("STEP 4B")
    integ_vhf_arr = np.full(vtec_vhf.shape[0], float(integ_vhf), dtype=float)
    Ne_rec_weighted, weights = step4B_weighted_reconstruct(
        D_all, vtec_all, integ_vhf_arr, integ_hf_per_ray, bw_vhf=BW_VHF, bw_hf=BW_HF,
        n_lat=len(lats), n_alt=len(alts_m), n_iters=50, relax=0.2
    )

    # Original ionosphere
    _print_ne_stats("Original Ne (iono)", iono if 'iono' in locals() else np.array([]))
    # Reconstructed (unweighted ART)
    _print_ne_stats("Reconstructed Ne (ART all)", Ne_rec_all if 'Ne_rec_all' in locals() else np.array([]))
    # Reconstructed (weighted ART)
    _print_ne_stats("Reconstructed Ne (weighted)", Ne_rec_weighted if 'Ne_rec_weighted' in locals() else np.array([]))
    # Optional: per-branch reconstructions if available
    if 'Ne_rec_vhf' in locals():
        _print_ne_stats("Reconstructed Ne (VHF ART)", Ne_rec_vhf)
    if 'Ne_rec_hf' in locals():
        _print_ne_stats("Reconstructed Ne (HF ART)", Ne_rec_hf)

    print("STEP 4C")
    # Weighted ART (use chunked ART if D_all is too tall)
    if sp.isspmatrix_csr(D_all) and D_all.shape[0] > MAX_ROWS_CHUNKED:
        sw = np.sqrt(np.maximum(weights, 0.0))
        if sp.issparse(D_all):
            Ds = D_all.multiply(sw[:, None])
        else:
            Ds = D_all * sw[:, None]
        ys = vtec_all * sw
        Ne_rec_weighted_hist, rmse_hist_w = art_kaczmarz_chunked(
            Ds, ys, n_lat=len(lats), n_alt=len(alts_m),
            n_iters=N_ITERS, relax=RELAX,
            rows_per_chunk=4000, nonneg=True, dtype=np.float32
        )
        reg_hist_w = None
    else:
        Ne_rec_weighted_hist, rmse_hist_w, reg_hist_w = weighted_art_with_history(
            D_all, vtec_all, weights,
            n_lat=len(lats), n_alt=len(alts_m),
            n_iters=N_ITERS, relax=RELAX,
            compute_lcurve=True, L=None
        )
    gc.collect(); free_all_gpu()

    print("STEP 4D")
    # Delta (Δt) branch (same behavior as your script)
    Ne_rec_ddt = delta_dt_branch(delta_t_vhf, delta_t_hf, D_hf, lats, alts_m,
                                 f_c_hf=F_C_HF, bw_hf=BW_HF,
                                 n_iters=N_ITERS, relax=RELAX)
    _print_ne_stats("Reconstructed Ne (delta_t branch)", Ne_rec_ddt if 'Ne_rec_ddt' in locals() else np.array([]))
    gc.collect(); free_all_gpu()

    # Ensure arrays are (n_alt, n_lat) for cross-sections (transpose if yours are (n_lat,n_alt))
    truth_alt_lat = iono
    rec_all_alt_lat = Ne_rec_all
    rec_w_alt_lat   = Ne_rec_weighted

    if STEP_6_PLOT_RECONS:
        print("Step 6: Plotting & Evaluation")
        plot_recons(
            s_no, lats, alts_m,
            [Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, Ne_rec_weighted, Ne_rec_ddt],
            ["VHF only", "HF only", "Combined (standard)", "Combined (weighted)", "δ(Δt)-based"]
        )

    return lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf

if __name__ == '__main__':

    
    # Introduce relation between integration time and incidence angle for HF rays (T_i = ((2*max_altitude)/sc_vel)*tan(theta_i))
    # Constrain the integration time, incidence angle and sat_lats_hf points because of this

    TOTAL_LAT_DEG = 20.0   # degrees of latitude covered by the transect (e.g. -10..+10)
    # Convert latitude span to along-track kilometers at satellite radius (R_EUROPA + SC_ALTITUDE)
    r_s_km = (R_EUROPA / 1e3) + (SC_ALTITUDE / 1e3)
    TOTAL_KM = (np.pi / 180.0) * r_s_km * TOTAL_LAT_DEG
    print("Total transect length (km):", TOTAL_KM)
    
    # Start latitude is the left edge of the span (center the span around 0° by default)
    LAT_START = -10
    VHF_INTEG_TIME = 0.12  # seconds
    VHF_resolution_km = (SC_VELOCITY * VHF_INTEG_TIME / 1e3) * (180.0/np.pi) / r_s_km  # degrees
    print("VHF resolution (deg):", VHF_resolution_km)
    SAT_LATS_VHF = np.arange(-10, 10, VHF_resolution_km)

    # # Case 1: coarse angles
    # theta_grid_coarse = np.linspace(2, 20, 10)

    # # Case 2: finer angles
    # theta_grid_fine = np.linspace(2, 20, 25)

    # # Build the transects (MODE A)
    # sat_lats_1, thetas_1, T_1 = plan_transect_fixed_length(
    #     theta_grid_deg=theta_grid_coarse,
    #     total_km=TOTAL_KM,
    #     lat_start_deg=LAT_START,
    #     R_E_km=R_EUROPA/1e3,       
    #     h_km=SC_ALTITUDE/1e3,
    #     v_km_s=SC_VELOCITY/1e3
    # )

    # print("Case 1: coarse angles")
    # print("theta_grid_coarse:", theta_grid_coarse)
    # print("T_1:", T_1)
    # print("sat_lats_1:", sat_lats_1)

    # sat_lats_2, thetas_2, T_2 = plan_transect_fixed_length(
    #     theta_grid_deg=theta_grid_fine,
    #     total_km=TOTAL_KM,
    #     lat_start_deg=LAT_START,
    #     R_E_km=R_EUROPA/1e3,
    #     h_km=SC_ALTITUDE/1e3,
    #     v_km_s=SC_VELOCITY/1e3
    # )

    # print("Case 2: fine angles")
    # print("theta_grid_fine:", theta_grid_fine)
    # print("T_2:", T_2)
    # print("sat_lats_2:", sat_lats_2)

    # # Feed into your pipeline — note: incidence_angle_deg and integ_hf now vary per shot
    # lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, dt_vhf, dt_hf = run_pipeline(
    #     s_no=1,
    #     enhancement_amplitude=5e9,
    #     sat_lats_vhf=SAT_LATS_VHF,
    #     npts_vhf=500,
    #     sat_lats_hf=sat_lats_2,
    #     incidence_angle_deg=thetas_2,
    #     npts_hf=500,
    #     integ_vhf=0.12,
    #     integ_hf=T_2
    # )
    # lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, dt_vhf, dt_hf = run_pipeline(
    #     s_no=3,
    #     enhancement_amplitude=5e9,
    #     sat_lats_vhf=SAT_LATS_VHF,
    #     npts_vhf=500,
    #     sat_lats_hf=sat_lats_1,
    #     incidence_angle_deg=thetas_1,
    #     npts_hf=500,
    #     integ_vhf=0.12,
    #     integ_hf=T_1
    # )

    # ---- user-set knobs ----
    ANGLES_TO_TEST = [0.05,0.1]
    LAT_START      = -10.0

    Rk = R_EUROPA/1e3
    hk = SC_ALTITUDE/1e3
    vk = SC_VELOCITY/1e3

    for idx, th in enumerate(ANGLES_TO_TEST, 1):
        # Build a fixed-length flight line with the *same* theta for every shot
        sat_lats_hf, theta_per_ray, integ_hf_per_ray = plan_transect_single_angle(
            theta_deg=th, total_km=TOTAL_KM, lat_start_deg=LAT_START,
            R_E_km=Rk, h_km=hk, v_km_s=vk, min_shots=2, clamp_last=True
        )

        # Run the full forward + invert pipeline
        lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, dt_vhf, dt_hf = run_pipeline(
            s_no=idx,
            enhancement_amplitude=5e9,
            sat_lats_vhf=SAT_LATS_VHF,    # your existing VHF grid
            npts_vhf=500,
            sat_lats_hf=sat_lats_hf,      # N+1
            incidence_angle_deg=theta_per_ray,   # length N
            npts_hf=500,
            integ_vhf=0.12,
            integ_hf=integ_hf_per_ray      # length N
        )

        print(f"\n[θ={th:.1f}°] shots={len(theta_per_ray)}, "
            f"T̄={np.mean(integ_hf_per_ray):.3f}s, lat span={sat_lats_hf[-1]-sat_lats_hf[0]:.3f}°")


    theta_grid = np.linspace(2, 20, 25)
    theta_scaled, T_scaled = scale_angles_to_match_length(
        theta_grid_deg=theta_grid,
        total_km=TOTAL_KM,
        h_km=SC_ALTITUDE/1e3,
        v_km_s=SC_VELOCITY/1e3
    )

    sat_lats_sameN = sat_lats_from_times(
        T_scaled, lat_start_deg=LAT_START,
        R_E_km=R_EUROPA/1e3, h_km=SC_ALTITUDE/1e3, v_km_s=SC_VELOCITY/1e3
    )

    lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, dt_vhf, dt_hf = run_pipeline(
        s_no=1,
        enhancement_amplitude=5e9,
        sat_lats_vhf=SAT_LATS_VHF,
        npts_vhf=500,
        sat_lats_hf=sat_lats_sameN,
        incidence_angle_deg=theta_scaled,
        npts_hf=500,
        integ_vhf=0.12,
        integ_hf=T_scaled
    )
    print("Case 1: scaled angles to match length and number of shots")
    print("theta_scaled:", theta_scaled)
    print("T_scaled:", T_scaled)
    print("sat_lats_sameN:", sat_lats_sameN)