import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# All angles used in this file should be interpreted as true incidence angle (from vertical, 0 = nadir, 90 = horizontal)

# Create a background ionosphere from your dataframe (altitude, electron density ne)
def build_ionosphere(df, lat_extent=(-10, 10), lat_res=100):
    altitude = df['altitude'].values  # Altitude array (km)
    ne_profile = df['ne'].values      # Electron density array (electrons/m^3)

    # Latitude extent (degrees)
    latitudes = np.linspace(lat_extent[0], lat_extent[1], lat_res)

    # Extrude electron density vertically across latitude
    ionosphere_map = np.tile(ne_profile[:, np.newaxis], (1, lat_res))

    return latitudes, altitude, ionosphere_map

# Insert Gaussian "enhancements" from another profile or defined parameters
def add_gaussian_enhancement(ionosphere_map, latitudes, altitude, 
                             lat_center=0, alt_center=300, 
                             lat_width=1, alt_width=20, amplitude=1e11):
    # Create meshgrid for the ionosphere map
    LAT, ALT = np.meshgrid(latitudes, altitude)

    # Gaussian function definition
    gauss = amplitude * np.exp(-((LAT - lat_center)**2 / (2 * lat_width**2) +
                                 (ALT - alt_center)**2 / (2 * alt_width**2)))

    # Add Gaussian bump
    enhanced_ionosphere = ionosphere_map + gauss

    return enhanced_ionosphere

# Plot example:
def plot_ionosphere(latitudes, altitude, ionosphere_map):
    plt.figure(figsize=(8, 6))
    plt.contourf(latitudes, altitude, ionosphere_map, levels=50, cmap='viridis')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Altitude (km)')
    plt.title('Electron Density (electrons/m³)')
    plt.colorbar(label='Electron Density')
    plt.gca()#.invert_yaxis()
    plt.show()

# Helper to find Gaussian parameters from avg profiles
def fit_gaussian_params(ne_profile, altitude):
    peak_idx = np.argmax(ne_profile)
    peak_alt = altitude[peak_idx]
    peak_ne = ne_profile[peak_idx]

    # Rough width estimate (standard deviation)
    half_max = peak_ne / 2
    indices_above_half = np.where(ne_profile > half_max)[0]
    if len(indices_above_half) > 1:
        alt_width = (altitude[indices_above_half[-1]] - altitude[indices_above_half[0]]) / 2.355  # FWHM to sigma
    else:
        alt_width = 20  # default guess if very narrow

    return peak_alt, alt_width, peak_ne