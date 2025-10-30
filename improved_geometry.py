# Fixed implementation for integration time and geometry matrix construction

import numpy as np
from scipy import sparse as sp


def build_geometry_matrix_weighted_sparse(rays, lats, alts_m, dtype=np.float32):
    """
    Build a SPARSE geometry matrix (CSR) with path-length weighting.
    Each ray segment contributes its ds to the pixel containing the segment midpoint.
    Returns a scipy.sparse.csr_matrix of shape (n_rays, n_lat*n_alt), dtype float32 by default.
    """
    from scipy.sparse import coo_matrix

    n_lat, n_alt = len(lats), len(alts_m)
    n_pix = n_lat * n_alt
    n_rays = len(rays)

    R_E = 1569e3  # Europa radius (m) -- keep your body radius
    deg2m = (np.pi/180.0) * R_E

    rows = []
    cols = []
    data = []

    for i, ray in enumerate(rays):
        # ray: (n_points, 2) = [lat_deg, alt_m]
        lat = ray[:, 0]; alt = ray[:, 1]
        for j in range(len(ray) - 1):
            lat0, alt0 = lat[j],   alt[j]
            lat1, alt1 = lat[j+1], alt[j+1]

            # midpoint
            lat_mid = 0.5*(lat0 + lat1)
            alt_mid = 0.5*(alt0 + alt1)

            # nearest pixel
            jφ = int(np.argmin(np.abs(lats    - lat_mid)))
            jh = int(np.argmin(np.abs(alts_m  - alt_mid)))

            if 0 <= jφ < n_lat and 0 <= jh < n_alt:
                # segment length (meters)
                dlat_m = (lat1 - lat0) * deg2m
                dalt_m = (alt1 - alt0)
                ds = float(np.hypot(dlat_m, dalt_m))

                p = jh * n_lat + jφ  # pixel index
                rows.append(i)
                cols.append(p)
                data.append(ds)

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    data = np.asarray(data, dtype=dtype)

    D_coo = coo_matrix((data, (rows, cols)), shape=(n_rays, n_pix), dtype=dtype)
    D_csr = D_coo.tocsr()         # sums duplicates per (i,p)
    D_csr.sort_indices()
    return D_csr

def build_geometry_matrix_weighted(rays, lats, alts_m):
    """
    Build geometry matrix with proper path-length weighting.
    This function is compatible with the main tomography script and does not require changes.
    
    Parameters:
    -----------
    rays : list of np.ndarray
        Ray paths as (n_points, 2) arrays of [lat, alt]
    lats : np.ndarray
        Latitude grid (degrees)
    alts_m : np.ndarray  
        Altitude grid (meters)
        
    Returns:
    --------
    D : np.ndarray
        Geometry matrix (n_rays, n_pixels) with path-length weights
    """
    n_lat, n_alt = len(lats), len(alts_m)
    n_pix = n_lat * n_alt
    n_rays = len(rays)
    
    D = np.zeros((n_rays, n_pix))
    R_E = 1569e3  # Europa radius in m
    deg2m = (np.pi/180.0) * R_E
    
    for i, ray in enumerate(rays):
        # Process each segment of the ray path
        for j in range(len(ray) - 1):
            lat0, alt0 = ray[j]
            lat1, alt1 = ray[j + 1]
            
            # Find grid cell indices for the segment midpoint
            lat_mid = (lat0 + lat1) / 2.0
            alt_mid = (alt0 + alt1) / 2.0
            
            jφ = int(np.argmin(np.abs(lats - lat_mid)))
            jh = int(np.argmin(np.abs(alts_m - alt_mid)))
            
            # Calculate path length through this segment
            dlat = (lat1 - lat0) * deg2m
            dalt = alt1 - alt0
            path_length = np.hypot(dlat, dalt)
            
            # Add weighted contribution to geometry matrix
            if 0 <= jφ < n_lat and 0 <= jh < n_alt:
                pixel_index = jh * n_lat + jφ
                D[i, pixel_index] += path_length
                
    return D

def weighted_reconstruction_art_sparse(D, y, weights, n_lat, n_alt, n_iters=50, relax=0.2):
    """
    Weighted variant that mirrors your previous logic:
    denom = w_i * (d_i·d_i), update uses w_i in numerator too (numerically cancels).
    """
    assert sp.issparse(D)
    D = D.tocsr()
    n_rays, n_pix = D.shape
    x = np.zeros(n_pix, dtype=np.float32)

    y = np.asarray(y, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)

    for _ in range(n_iters):
        for i in range(n_rays):
            row = D.getrow(i)
            idx = row.indices
            dat = row.data
            if dat.size == 0:
                continue
            proj  = (dat * x[idx]).sum()
            denom = w[i] * (dat * dat).sum()      # matches your earlier form
            if denom <= 0:
                continue
            resid = y[i] - proj
            coef  = relax * w[i] * (resid / denom)
            x[idx] += coef * dat
    return x.reshape((n_alt, n_lat))

def weighted_reconstruction_art(D, vtec_measurements, measurement_weights, 
                               n_lat, n_alt, n_iters=50, relax=0.2):
    """
    ART reconstruction with measurement weighting to account for different noise levels.
    
    Parameters:
    -----------
    D : np.ndarray
        Geometry matrix (n_rays, n_pixels)
    vtec_measurements : np.ndarray
        VTEC measurements from all ray types
    measurement_weights : np.ndarray
        Weights for each measurement (higher = more reliable)
    n_lat, n_alt : int
        Grid dimensions
    n_iters : int
        Number of ART iterations
    relax : float
        Relaxation parameter
        
    Returns:
    --------
    Ne_reconstructed : np.ndarray
        Reconstructed electron density (n_alt, n_lat)
    """
    n_rays, n_pix = D.shape
    Ne = np.zeros(n_pix)
    
    for iteration in range(n_iters):
        for i in range(n_rays):
            Di = D[i]
            weight = measurement_weights[i]
            
            # Current projection
            proj = Di @ Ne
            
            # Weighted denominator
            denom = weight * (Di @ Di)
            
            if denom > 0:
                residual = vtec_measurements[i] - proj
                # Apply weight to the update
                Ne += relax * weight * residual / denom * Di
    
    return Ne.reshape((n_alt, n_lat))

def calculate_measurement_weights(integration_times, bandwidths, snr_values=None):
    """
    Calculate measurement weights based on integration time and bandwidth.
    Higher integration time and bandwidth = lower noise = higher weight.
    
    Parameters:
    -----------
    integration_times : np.ndarray
        Integration time for each measurement (seconds)
    bandwidths : np.ndarray
        Bandwidth for each measurement (Hz)
    snr_values : np.ndarray, optional
        SNR for each measurement (linear scale)
        
    Returns:
    --------
    weights : np.ndarray
        Normalized weights (higher = more reliable measurement)
    """
    if snr_values is None:
        snr_values = np.ones_like(integration_times)
    
    # Weight proportional to sqrt(BW * T * SNR) - from noise theory
    weights = np.sqrt(bandwidths * integration_times * snr_values)
    
    # Normalize weights to have mean = 1
    weights = weights / np.mean(weights)
    
    return weights

# ===================================================================
# MODIFIED MAIN CODE SECTIONS
# ===================================================================

def improved_main_reconstruction():
    """
    Main reconstruction with consistent integration times and weighted geometry.
    """
    # Load your ionosphere setup (keeping your existing code)
    # ... [your ionosphere setup code] ...
    
    # ===== SOLUTION 1: CONSISTENT INTEGRATION TIMES =====
    
    # Option A: Use same integration time for all measurements
    CONSISTENT_INTEGRATION_TIME = 0.1  # seconds
    
    # Option B: Use realistic varying times but account for them in weighting
    # VHF integration times (realistic: longer for weaker signals)
    integration_times_vhf = np.full(60, 0.12)  # 0.12s for all VHF
    
    # HF integration times (realistic: shorter for stronger signals at low angles)
    theta_list = [30, 40, 50, 60, 70, 80, 85]
    integration_times_hf_per_angle = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    
    # ===== VHF PROCESSING WITH CONSISTENT INTEGRATION =====
    
    # Build VHF rays (your existing code)
    sat_lats = np.linspace(-6.0, 6.0, 60)
    rays_vhf = trace_passive_nadir(sat_lats, alts_m, npts=500)
    
    # Compute STEC (your existing code)
    stec_vhf = np.array([
        compute_STEC_along_path(iono, lats, alts_m, ray)
        for ray in rays_vhf
    ])
    
    # VHF dual-frequency conversion with consistent integration time
    vtec_vhf = []
    delta_t_vhf = []
    
    for i, ray_stec in enumerate(stec_vhf):
        T = integration_times_vhf[i]  # Now consistent
        vtec_i, dt_i = dualfreq_to_VTEC(
            stec_slant=np.array([ray_stec]),
            f_c=60e6,
            bw=10e6,
            theta_deg=0.0,
            integration_time=T,
            return_deltat=True
        )
        vtec_vhf.append(vtec_i[0])
        delta_t_vhf.append(dt_i[0])
    
    vtec_vhf = np.array(vtec_vhf)
    delta_t_vhf = np.array(delta_t_vhf)
    
    # ===== IMPROVED VHF GEOMETRY MATRIX =====
    
    print("Building weighted VHF geometry matrix...")
    D_vhf = build_geometry_matrix_weighted(rays_vhf, lats, alts_m)
    
    # ===== HF PROCESSING WITH CONSISTENT INTEGRATION =====
    
    f_c_hf = 9e6
    bw_hf = 1e6
    
    all_rays_hf = []
    all_vtec_hf = []
    all_deltat_hf = []
    all_integration_times_hf = []
    
    for idx, θ in enumerate(theta_list):
        T_hf = integration_times_hf_per_angle[idx]  # Consistent per angle
        
        # Trace oblique rays
        rays_hf = trace_passive_oblique(
            sat_lats=sat_lats,
            h0=alts_m.max(),
            hsat=alts_m.max(),
            theta_i_deg=θ,
            npts=500  # Increased from 200 to match VHF resolution
        )
        
        # Compute STEC
        stec = np.array([
            compute_STEC_along_path(iono, lats, alts_m, ray)
            for ray in rays_hf
        ])
        
        # Dual-frequency conversion with consistent integration time per angle
        vtec, delta_t = dualfreq_to_VTEC(
            stec_slant=stec,
            f_c=f_c_hf,
            bw=bw_hf,
            theta_deg=θ,
            integration_time=T_hf,
            return_deltat=True
        )
        
        # Store results
        all_rays_hf.extend(rays_hf)
        all_vtec_hf.append(vtec)
        all_deltat_hf.append(delta_t)
        all_integration_times_hf.extend([T_hf] * len(rays_hf))
    
    # Flatten arrays
    all_vtec_hf = np.hstack(all_vtec_hf)
    all_deltat_hf = np.hstack(all_deltat_hf)
    all_integration_times_hf = np.array(all_integration_times_hf)
    
    # ===== IMPROVED HF GEOMETRY MATRIX =====
    
    print("Building weighted HF geometry matrix...")
    D_hf = build_geometry_matrix_weighted(all_rays_hf, lats, alts_m)
    
    # ===== WEIGHTED RECONSTRUCTION =====
    
    # Calculate measurement weights
    all_integration_times = np.concatenate([integration_times_vhf, all_integration_times_hf])
    all_bandwidths = np.concatenate([
        np.full(len(vtec_vhf), 10e6),  # VHF bandwidth
        np.full(len(all_vtec_hf), 1e6)  # HF bandwidth
    ])
    
    measurement_weights = calculate_measurement_weights(
        integration_times=all_integration_times,
        bandwidths=all_bandwidths
    )
    
    print(f"VHF weights (mean): {np.mean(measurement_weights[:len(vtec_vhf)]):.3f}")
    print(f"HF weights (mean): {np.mean(measurement_weights[len(vtec_vhf):]):.3f}")
    
    # Combine all measurements and geometry
    D_all = np.vstack([D_vhf, D_hf])
    vtec_all = np.concatenate([vtec_vhf, all_vtec_hf])
    
    # ===== RECONSTRUCTIONS =====
    
    n_iters = 50
    relax = 0.2
    
    # Standard ART reconstructions (for comparison)
    Ne_rec_vhf = reconstruct_art(D_vhf, vtec_vhf, n_lat, n_alt, n_iters, relax)
    Ne_rec_hf = reconstruct_art(D_hf, all_vtec_hf, n_lat, n_alt, n_iters, relax)
    Ne_rec_combined_standard = reconstruct_art(D_all, vtec_all, n_lat, n_alt, n_iters, relax)
    
    # Weighted ART reconstruction (NEW!)
    Ne_rec_combined_weighted = weighted_reconstruction_art(
        D_all, vtec_all, measurement_weights, n_lat, n_alt, n_iters, relax
    )
    
    return {
        'vhf_only': Ne_rec_vhf,
        'hf_only': Ne_rec_hf, 
        'combined_standard': Ne_rec_combined_standard,
        'combined_weighted': Ne_rec_combined_weighted,
        'weights': measurement_weights,
        'integration_times': all_integration_times
    }

# # Example usage in your main script:
# if __name__ == "__main__":
#     # Replace your reconstruction section with:
#     results = improved_main_reconstruction()
    
#     # Plot all four reconstructions
#     fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
#     titles = ["VHF only", "HF only", "Combined (standard)", "Combined (weighted)"]
#     recs = [results['vhf_only'], results['hf_only'], 
#             results['combined_standard'], results['combined_weighted']]
    
#     for ax, rec, title in zip(axs, recs, titles):
#         im = ax.pcolormesh(lats, alts_m/1e3, rec*1e-6,
#                           shading='auto', cmap='viridis')
#         ax.set_title(title)
#         ax.set_xlabel("Latitude (°)")
#         if ax is axs[0]:
#             ax.set_ylabel("Altitude (km)")
#         fig.colorbar(im, ax=ax, label='Ne (×10⁶ cm⁻³)')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print weight statistics
#     n_vhf = 60
#     print(f"\nMeasurement weight statistics:")
#     print(f"VHF weights: {results['weights'][:n_vhf].mean():.3f} ± {results['weights'][:n_vhf].std():.3f}")
#     print(f"HF weights: {results['weights'][n_vhf:].mean():.3f} ± {results['weights'][n_vhf:].std():.3f}")
#     print(f"Weight ratio (VHF/HF): {results['weights'][:n_vhf].mean() / results['weights'][n_vhf:].mean():.2f}")

# All theta/angle variables should be interpreted as true incidence angle (from vertical, 0 = nadir, 90 = horizontal)