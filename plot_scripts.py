import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
import os

# Save Location
IMG_TS = datetime.now().strftime("%m_%d_%H_%M_%S")
IMG_DIR = os.path.join("img", IMG_TS)
os.makedirs(IMG_DIR, exist_ok=True)

def plot_ionosphere_and_enhancement(s_no, latitudes, altitude, ionosphere_map, enhanced_ionosphere):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Original ionosphere
    c1 = axs[0].contourf(latitudes, altitude, ionosphere_map, levels=50, cmap='viridis')
    axs[0].set_xlabel('Latitude (degrees)')
    axs[0].set_ylabel('Altitude (km)')
    axs[0].set_title('Original Electron Density (electrons/m³)')
    fig.colorbar(c1, ax=axs[0], label='Electron Density')

    # Enhanced ionosphere
    c2 = axs[1].contourf(latitudes, altitude, enhanced_ionosphere, levels=50, cmap='viridis')
    axs[1].set_xlabel('Latitude (degrees)')
    axs[1].set_ylabel('Altitude (km)')
    axs[1].set_title('Enhanced Electron Density (electrons/m³)')
    fig.colorbar(c2, ax=axs[1], label='Electron Density')

    plt.tight_layout()
    # plt.show(block=False)
    plt.savefig(os.path.join(IMG_DIR, f"A_ionosphere_{s_no}.png"))

def plot_raypaths(s_no, lats, alts_m, rays_hf, theta_per_ray, rays_vhf,
                  title_hf="Modeled Raypaths (HF)",
                  title_vhf="Modeled Raypaths (VHF)",
                  max_rays_per_angle=None):
    """
    Two-panel figure:
      left  = HF oblique rays, color-coded by true incidence angle (deg from vertical)
      right = VHF nadir rays (vertical)
    """

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
    # plt.show(block=False)
    plt.savefig(os.path.join(IMG_DIR, f"B_gpu_raypaths_{s_no}.png"))

def plot_recons(s_no, lats, alts_m, recs, titles):
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
    plt.tight_layout(); plt.show(block=False); plt.savefig(os.path.join(IMG_DIR, f"A_gpu_reconstructions_{s_no}.png"))