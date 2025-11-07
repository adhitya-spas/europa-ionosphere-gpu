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
from plot_scripts import plot_raypaths, plot_ionosphere_and_enhancement, plot_recons
USE_GPU = True
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import scipy.sparse as sp
except Exception:
    USE_GPU = False
    cp = None
    cpx_sparse = None
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


    # Build Geometry Matrices
    # Use the latitude grid `lats` (not satellite lat arrays) so columns match ionosphere pixels
    D_vhf = build_geometry_matrix_weighted_sparse(rays_vhf, lats, alts_m, dtype=np.float32)
    D_hf  = build_geometry_matrix_weighted_sparse(rays_hf,  lats, alts_m, dtype=np.float32)
    D_all = scipy.sparse.vstack([D_vhf, D_hf], format="csr")
    # D_all_gpu = cpx_sparse.csr_matrix(D_all)

    # HF path (GPU)
    # Push ionosphere to GPU and validate shapes
    iono_gpu = cp.asarray(iono, dtype=cp.float32)          # (n_alt, n_lat)
    x_gpu = iono_gpu.ravel(order="C")                      # (n_pix,)
    D_hf_gpu = cpx_sparse.csr_matrix(D_hf)
    # Validate expected dimensions before calling cuSPARSE
    n_rows, n_cols = D_hf.shape
    if x_gpu.size != n_cols:
        print(f"Dimension mismatch BEFORE GPU multiply: D_hf.shape={D_hf.shape}, x_gpu.shape={x_gpu.shape}")
        raise ValueError(f"Geometry columns ({n_cols}) != ionosphere pixels ({x_gpu.size})")
    # TEC truth (HF; GPU)
    try:
        tec_true_gpu = D_hf_gpu @ x_gpu                        # (N,)
    except Exception:
        print("Error during GPU sparse multiply:")
        print(f" D_hf_gpu.shape = {D_hf_gpu.shape}, D_hf.dtype={D_hf.dtype}, nnz={D_hf.nnz}")
        print(f" x_gpu.shape = {x_gpu.shape}, x_gpu.dtype={x_gpu.dtype}")
        raise
    tec_true = cp.asnumpy(tec_true_gpu)                    # back to CPU for helper
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

    return rays_vhf, rays_hf, D_all, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf, D_vhf, D_hf

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

    # ---- Push ionosphere to GPU and vectorize ----
    iono_gpu = cp.asarray(iono, dtype=cp.float32)          # (n_alt, n_lat)
    x_gpu = iono_gpu.ravel(order="C")                      # (n_pix,)

    assert D_hf_gpu.shape[0] == tec_true_gpu.shape[0]

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

# ---------------------- Step 4: BUILD GEOMETRY MATRICES -----------------------
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

# ---------------------- Step 0: Correlation between theta and integration time -----------------------

def plan_transect_fixed_length(theta_grid_deg, total_km,
                               lat_start_deg, R_E_km, h_km, v_km_s):
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
    K   = (2.0 * h_km) / v_km_s                 # seconds per tan(theta)
    sat_lats = [float(lat_start_deg)]
    theta_list = []
    T_list = []
    dist_accum = 0.0
    i = 0

    while dist_accum < total_km:
        theta = float(theta_grid_deg[i % len(theta_grid_deg)])
        Ti = K * np.tan(np.radians(theta))      # seconds
        di = v_km_s * Ti                        # km advanced this shot
        dphi_deg = (di / r_s) * (180.0 / np.pi)

        # Stop if the next step would overshoot too much; otherwise take it
        if dist_accum + di > total_km:
            # take a partial last step so the total is exact (optional)
            remain = total_km - dist_accum
            frac = remain / di if di > 0 else 0.0
            sat_lats.append(sat_lats[-1] + frac * dphi_deg)
            theta_list.append(theta)
            T_list.append(frac * Ti)            # shorten last integration proportionally
            break

        sat_lats.append(sat_lats[-1] + dphi_deg)
        theta_list.append(theta)
        T_list.append(Ti)
        dist_accum += di
        i += 1

    return np.asarray(sat_lats), np.asarray(theta_list), np.asarray(T_list)


def scale_angles_to_match_length(theta_grid_deg, total_km, h_km, v_km_s):
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
    T_new = K * tan_new  # since tan_new already computed
    return theta_new, T_new


# ---------------------- MAIN PIPELINE ----------------------
def run_pipeline(s_no,
                 enhancement_amplitude,
                 sat_lats_vhf, npts_vhf,
                 sat_lats_hf, incidence_angle_deg, npts_hf,
                 integ_vhf, integ_hf):
    # STEP 1: Load ionosphere and build ionosphere map
    lats, alt_km, iono = load_dataframe_and_build_ionosphere(s_no, enhancement_amplitude)

    # STEP 2: Generate ray paths (GPU)
    rays_vhf, rays_hf, D_all, vtec_vhf, vtec_hf, delta_t_vhf, delta_t_hf, D_vhf, D_hf = generate_ray_paths(
        s_no=s_no,
        sat_lats_vhf=sat_lats_vhf,
        alts_m=alt_km*1e3,
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
    print("incidence_angle_deg length:", len(incidence_angle_deg))
    print("integ_hf length:", len(integ_hf))
    assert D_hf.shape[0] == len(incidence_angle_deg) == len(integ_hf)

    # STEP 3: Forward model (GPU)
    tec_true_hf, vtec_hf_gpu, delta_t_hf_gpu = forward_model_gpu(
        iono=iono,
        lats=lats,
        alts_m=alt_km*1e3,
        D_hf=D_hf,
        theta_per_ray=incidence_angle_deg,
        integ_hf=integ_hf
    )

    # STEP 4: RECONSTRUCTION 
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

    # Step 4B (Weighted ART)
    Ne_rec_weighted, weights = step4B_weighted_reconstruct(
        D_all, vtec_all, integ_vhf, integ_hf, bw_vhf=BW_VHF, bw_hf=BW_HF,
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

    Ne_rec_weighted_hist, rmse_hist_w, reg_hist_w = weighted_art_with_history(
        D_all, vtec_all, weights,
        n_lat=len(lats), n_alt=len(alts_m),
        n_iters=N_ITERS, relax=RELAX,
        compute_lcurve=True, L=None
    )

    # Optional δ(Δt) branch (same behavior as your script)
    Ne_rec_ddt = delta_dt_branch(delta_t_vhf, delta_t_hf, D_hf, lats, alts_m,
                                 f_c_hf=F_C_HF, bw_hf=BW_HF,
                                 n_iters=N_ITERS, relax=RELAX)
    _print_ne_stats("Reconstructed Ne (delta_t branch)", Ne_rec_ddt if 'Ne_rec_ddt' in locals() else np.array([]))

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
    # Choose a total along-track length in km (same for all cases)
    TOTAL_KM = 500.0    # example: 500 km transect for each case
    LAT_START = -5.0

    # Case 1: coarse angles
    theta_grid_coarse = np.linspace(2, 20, 10)

    # Case 2: finer angles
    theta_grid_fine = np.linspace(2, 20, 25)

    # Build the transects (MODE A)
    sat_lats_1, thetas_1, T_1 = plan_transect_fixed_length(
        theta_grid_deg=theta_grid_coarse,
        total_km=TOTAL_KM,
        lat_start_deg=LAT_START,
        R_E_km=R_EUROPA/1e3,       
        h_km=SC_ALTITUDE/1e3,
        v_km_s=SC_VELOCITY/1e3
    )

    sat_lats_2, thetas_2, T_2 = plan_transect_fixed_length(
        theta_grid_deg=theta_grid_fine,
        total_km=TOTAL_KM,
        lat_start_deg=LAT_START,
        R_E_km=R_EUROPA/1e3,
        h_km=SC_ALTITUDE/1e3,
        v_km_s=SC_VELOCITY/1e3
    )

    # Feed into your pipeline — note: incidence_angle_deg and integ_hf now vary per shot
    lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, dt_vhf, dt_hf = run_pipeline(
        s_no=1,
        enhancement_amplitude=5e9,
        sat_lats_vhf=np.linspace(-5.0, 5.0, 50),
        npts_vhf=500,
        sat_lats_hf=sat_lats_2,                 # <-- per-shot lats for the HF transect
        incidence_angle_deg=thetas_2,           # <-- per-shot θ
        npts_hf=500,
        integ_vhf=0.12,
        integ_hf=T_2                            # <-- per-shot Δt (same length as θ)
    )
    lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, dt_vhf, dt_hf = run_pipeline(
        s_no=3,
        enhancement_amplitude=5e9,
        sat_lats_vhf=np.linspace(-5.0, 5.0, 50),
        npts_vhf=500,
        sat_lats_hf=sat_lats_1,                 # <-- per-shot lats for the HF transect
        incidence_angle_deg=thetas_1,           # <-- per-shot θ
        npts_hf=500,
        integ_vhf=0.12,
        integ_hf=T_1                            # <-- per-shot Δt (same length as θ)
    )

    theta_grid = np.linspace(2, 20, 25)   # desired N
    theta_scaled, T_scaled = scale_angles_to_match_length(
        theta_grid_deg=theta_grid,
        total_km=TOTAL_KM,
        h_km=SC_ALTITUDE/1e3,
        v_km_s=SC_VELOCITY/1e3
    )

    # Now build SAT_LATS_HF from the per-shot times (so geometry matches motion)
    def sat_lats_from_times(T_list, lat_start_deg, R_E_km, h_km, v_km_s):
        r_s = R_E_km + h_km
        d_km = v_km_s * np.asarray(T_list)
        dphi = (d_km / r_s) * (180.0/np.pi)
        lats = [lat_start_deg]
        for step in dphi:
            lats.append(lats[-1] + step)
        return np.asarray(lats)

    sat_lats_sameN = sat_lats_from_times(
        T_scaled, lat_start_deg=LAT_START,
        R_E_km=R_EUROPA/1e3, h_km=SC_ALTITUDE/1e3, v_km_s=SC_VELOCITY/1e3
    )

    # Run
    lats, alt_km, iono, D_all, vtec_vhf, vtec_hf, dt_vhf, dt_hf = run_pipeline(
        s_no=2,
        enhancement_amplitude=5e9,
        sat_lats_vhf=np.linspace(-5.0, 5.0, 50),
        npts_vhf=500,
        sat_lats_hf=sat_lats_sameN,      # len = N+1
        incidence_angle_deg=theta_scaled, # len = N
        npts_hf=500,
        integ_vhf=0.12,
        integ_hf=T_scaled                 # len = N
    )

    # run_pipeline(
    #     s_no=1,
    #     enhancement_amplitude=5e9,
    #     sat_lats_vhf=np.linspace(-5.0, 5.0, 50),
    #     npts_vhf=500,
    #     sat_lats_hf=np.linspace(-5.0, 5.0, 50),
    #     incidence_angle_deg=np.linspace(1, 20, 20),
    #     npts_hf=500,
    #     integ_vhf=0.12,
    #     integ_hf=0.1 
    # )