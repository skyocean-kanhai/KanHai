"""
Ocean Data Loader Module

This module provides functions to load and preprocess oceanographic data for machine learning inference.
It handles surface data (SLA, SST, SSS, velocity) and subsurface data (temperature, salinity, velocity at multiple depths).
"""

from typing import List, Tuple

import numpy as np
import xarray as xr


def load_single_day_input(
        surface_file: str,
        sst_file: str,
        sss_file: str,
        depth_file: str,
        is_normalized: bool = False,
        ocean_mean: np.ndarray = None,
        ocean_var: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load single-day ocean input data for inference or testing (without labels).
    
    This function loads and preprocesses oceanographic data from multiple sources:
    - Surface data: Sea Level Anomaly (SLA), Sea Surface Temperature (SST), 
      Sea Surface Salinity (SSS), and velocity components (u, v)
    - Subsurface data: Temperature, salinity, and velocity at multiple depth levels
    
    Args:
        surface_file (str): Path to surface data file (SLA and velocity)
        sst_file (str): Path to Sea Surface Temperature file
        sss_file (str): Path to Sea Surface Salinity file  
        depth_file (str): Path to subsurface ocean data file
        is_normalized (bool): Whether to apply normalization using provided statistics
        ocean_mean (np.ndarray): Mean values for normalization (85 channels)
        ocean_var (np.ndarray): Standard deviation values for normalization (85 channels)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Processed input data array (85, 400, 480)
            - Mask array for handling NaN values
    """

    # Load surface data: SLA, SST, SSS, and velocity components (u, v)
    # Note: Different longitude/latitude ranges for different data sources
    
    # Load Sea Surface Temperature (SST) data
    sur_sst = xr.open_dataset(sst_file)[["sst"]].sel(
        longitude=slice(100, 159.875), latitude=slice(0, 49.875)
    ).to_array().values

    # Load Sea Surface Salinity (SSS) data
    sur_so = xr.open_dataset(sss_file)[['sss_output']].sel(
        longitude=slice(100, 159.875), latitude=slice(0, 49.875)
    ).to_array().values

    # Load Sea Level Anomaly (SLA) data
    sur_sla = xr.open_dataset(surface_file)[['sla']].sel(
        longitude=slice(100, 160), latitude=slice(0, 50)
    ).to_array().values

    # Load velocity components (u, v) from geostrophic currents
    sur_var = xr.open_dataset(surface_file)[['ugos', 'vgos']].sel(
        longitude=slice(100, 160), latitude=slice(0, 50)
    ).to_array().values

    # Handle missing data flags (commented out as they may not be needed)
    # sur_sla[sur_sla == -214748.3647] = np.nan
    # sur_var[sur_var == -214748.3647] = np.nan

    # Reshape data to standard format (channels, height, width)
    sur_sst = sur_sst.reshape(-1, sur_sst.shape[-2], sur_sst.shape[-1])
    sur_so = sur_so.reshape(-1, sur_so.shape[-2], sur_so.shape[-1])
    sur_sla = sur_sla.reshape(-1, sur_sla.shape[-2], sur_sla.shape[-1])
    sur_var = sur_var.reshape(-1, sur_var.shape[-2], sur_var.shape[-1])

    # Concatenate all surface variables into a single array
    surface_data = np.concatenate((sur_sla, sur_sst, sur_so, sur_var), axis=0)

    # Load subsurface data at multiple depth levels
    # Variables: temperature (thetao), salinity (so), zonal velocity (uo), meridional velocity (vo)
    deep_vars_list = ['thetao', 'so', 'uo', 'vo']
    # Select specific depth levels (20 levels total)
    depth_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]

    depth_var = xr.open_dataset(depth_file)[deep_vars_list].sel(
        longitude=slice(100, 159.875), latitude=slice(0, 49.875)
    ).to_array().values

    # Reshape subsurface data to standard format
    depth_var = depth_var.reshape(-1, depth_var.shape[-2], depth_var.shape[-1])

    # Combine surface and subsurface data into final input array
    x = np.concatenate((surface_data, depth_var), axis=0).astype(np.float32)

    # Apply normalization if requested
    if is_normalized and ocean_mean is not None and ocean_var is not None:
        x = (x - ocean_mean.reshape(85, 1, 1)) / ocean_var.reshape(85, 1, 1)
    
    # Create mask for handling NaN values (using SLA and depth data)
    x_mask = np.concatenate((sur_sla, depth_var), axis=0).astype(np.float32)
    
    # Replace NaN values with zeros for model input
    x = np.nan_to_num(x)

    return x, x_mask


def load_single_day_recon_input(
        surface_file: str,
        sst_file: str,
        sss_file: str,
        depth_file: str,
        is_normalized: bool = False,
        ocean_mean: np.ndarray = None,
        ocean_var: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load single-day ocean input data for reconstruction model inference.
    
    This function is similar to load_single_day_input but specifically designed for the 
    reconstruction model, with slight differences in depth level selection and mask creation.
    
    Args:
        surface_file (str): Path to surface data file (SLA and velocity)
        sst_file (str): Path to Sea Surface Temperature file
        sss_file (str): Path to Sea Surface Salinity file  
        depth_file (str): Path to subsurface ocean data file
        is_normalized (bool): Whether to apply normalization using provided statistics
        ocean_mean (np.ndarray): Mean values for normalization (85 channels)
        ocean_var (np.ndarray): Standard deviation values for normalization (85 channels)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Processed input data array (85, 400, 480)
            - Mask array for handling NaN values (depth data only)
    """

    # Load surface data: SLA, SST, SSS, and velocity components (u, v)
    # Note: Different longitude/latitude ranges for different data sources
    
    # Load Sea Surface Temperature (SST) data
    sur_sst = xr.open_dataset(sst_file)[["sst"]].sel(
        longitude=slice(100, 159.875), latitude=slice(0, 49.875)
    ).to_array().values

    # Load Sea Surface Salinity (SSS) data
    sur_so = xr.open_dataset(sss_file)[['sss_output']].sel(
        longitude=slice(100, 159.875), latitude=slice(0, 49.875)
    ).to_array().values

    # Load Sea Level Anomaly (SLA) data
    sur_sla = xr.open_dataset(surface_file)[['sla']].sel(
        longitude=slice(100, 160), latitude=slice(0, 50)
    ).to_array().values

    # Load velocity components (u, v) from geostrophic currents
    sur_var = xr.open_dataset(surface_file)[['ugos', 'vgos']].sel(
        longitude=slice(100, 160), latitude=slice(0, 50)
    ).to_array().values

    # Handle missing data flags (commented out as they may not be needed)
    # sur_sla[sur_sla == -214748.3647] = np.nan
    # sur_var[sur_var == -214748.3647] = np.nan

    # Reshape data to standard format (channels, height, width)
    sur_sst = sur_sst.reshape(-1, sur_sst.shape[-2], sur_sst.shape[-1])
    sur_so = sur_so.reshape(-1, sur_so.shape[-2], sur_so.shape[-1])
    sur_sla = sur_sla.reshape(-1, sur_sla.shape[-2], sur_sla.shape[-1])
    sur_var = sur_var.reshape(-1, sur_var.shape[-2], sur_var.shape[-1])

    # Concatenate all surface variables into a single array
    surface_data = np.concatenate((sur_sla, sur_sst, sur_so, sur_var), axis=0)

    # Load subsurface data at multiple depth levels
    # Variables: temperature (thetao), salinity (so), zonal velocity (uo), meridional velocity (vo)
    deep_vars_list = ['thetao', 'so', 'uo', 'vo']
    # Select specific depth levels (20 levels total)
    depth_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]

    # Load subsurface data with specific depth level selection
    depth_var = xr.open_dataset(depth_file)[deep_vars_list].sel(
        longitude=slice(100, 159.875), latitude=slice(0, 49.875)
    ).isel(depth=depth_idx).to_array().values

    # Reshape subsurface data to standard format
    depth_var = depth_var.reshape(-1, depth_var.shape[-2], depth_var.shape[-1])

    # Combine surface and subsurface data into final input array
    x = np.concatenate((surface_data, depth_var), axis=0).astype(np.float32)

    # Apply normalization if requested
    if is_normalized and ocean_mean is not None and ocean_var is not None:
        x = (x - ocean_mean.reshape(85, 1, 1)) / ocean_var.reshape(85, 1, 1)
    
    # Create mask for handling NaN values (using depth data only for reconstruction)
    x_mask = depth_var.astype(np.float32)
    
    # Replace NaN values with zeros for model input
    x = np.nan_to_num(x)

    return x, x_mask