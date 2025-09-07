"""
Ocean Reconstruction Model Inference Script

This script performs inference using a trained ocean reconstruction model to generate
3D ocean state fields from surface observations and background subsurface data.

The reconstruction model takes surface data (SLA, SST, SSS, velocity) and background
subsurface data as input, and outputs reconstructed 3D ocean fields including
temperature, salinity, and velocity at multiple depth levels.
"""

import os
import re
import argparse
import onnxruntime
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime

from dataloader import load_single_day_recon_input

from datetime import date, timedelta


def date_series(last_n: int, length: int, ref_date: date = None):
    """
    Generate a date series relative to a reference date.
    
    Args:
        last_n (int): Number of days from the reference date to the last date
        length (int): Length of the date series
        ref_date (date): Reference date, defaults to today
        
    Returns:
        List[str]: List of date strings in YYYYMMDD format
    """
    if ref_date is None:
        ref_date = date.today()  # e.g., 2025-06-08
    start = ref_date - timedelta(days=last_n + length - 1)
    return [
        (start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(length)
    ]


def find_files_by_keyword_and_dates(root_folder, keyword_list, date_list):
    """
    Find files in a directory tree that match specific keywords and dates.
    
    Args:
        root_folder (str): Root directory to search
        keyword_list (List[str]): Keywords to match in filenames
        date_list (List[str]): List of dates in YYYYMMDD format
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping dates to file paths
    """
    result_dict = {}
    date_set = set(date_list)

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Match both YYYYMMDD and YYYY-MM-DD date formats
            match_yyyymmdd = re.search(r"(\d{8})", filename)
            match_yyyy_mm_dd = re.search(r"(\d{4}-\d{2}-\d{2})", filename)

            date_str = None
            if match_yyyymmdd:
                # Convert YYYYMMDD format to YYYY-MM-DD
                raw_date = match_yyyymmdd.group(1)
                date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            elif match_yyyy_mm_dd:
                # Use YYYY-MM-DD format directly
                date_str = match_yyyy_mm_dd.group(1)

            if date_str and any(k.lower() in filename.lower() for k in keyword_list):
                # Convert YYYY-MM-DD to YYYYMMDD for dictionary key
                date_key = date_str.replace("-", "")
                full_path = os.path.join(dirpath, filename)
                result_dict.setdefault(date_key, []).append(full_path)

    return result_dict


def npy2nc(npy_data):
    """
    Convert numpy array data to NetCDF format and save to file.
    
    Args:
        npy_data (np.ndarray): 3D ocean data array (80 channels, height, width)
    """
    try:
        y_file_path = os.path.join(src_dir, "template.nc")
        if not os.path.exists(y_file_path):
            raise FileNotFoundError(f"Template file not found: {y_file_path}")

        ds_template = xr.open_dataset(y_file_path)
        lon = ds_template['longitude'].sel(longitude=slice(100, 159.875))
        lat = ds_template['latitude'].sel(latitude=slice(0, 49.875))
        depth_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32]
        depth = xr.open_dataset(y_file_path)['depth'].isel(depth=depth_idx).values

        variables = ['thetao', 'so', 'uo', 'vo']
        layer_index = [list(range(0, 20)), list(range(20, 40)), list(range(40, 60)), list(range(60, 80))]
        array_list = []
        nc_file_single_dir_path = os.path.join(args.save_dir, datatime_str[0:4])
        os.makedirs(nc_file_single_dir_path, exist_ok=True)

        nc_file_path = os.path.join(nc_file_single_dir_path, f"recon_{datatime_str}.nc")
        
        for i in range(len(variables)):
            coords_dict = {
                'time': pd.date_range(datatime_str[:4] + "-" + datatime_str[4:6] + "-" + datatime_str[6:] + " 00:00:00",
                                      periods=1),
                'latitude': lat.astype(np.float32),
                'longitude': lon.astype(np.float32)
            }

            dims = ['time', 'depth', 'latitude', 'longitude']
            coords_dict['depth'] = depth.astype(np.float32)
            data = np.expand_dims(npy_data[layer_index[i], :, :], axis=0)

            nc_data_array = xr.DataArray(data.astype(np.float32), dims=dims, coords=coords_dict)
            array_list.append(nc_data_array)

        ds = xr.Dataset({
            'thetao': array_list[0],
            'so': array_list[1],
            'uo': array_list[2],
            'vo': array_list[3],
        })

        ds.to_netcdf(nc_file_path)
        print(f"✅ Successfully saved reconstruction results: {nc_file_path}")
    except Exception as e:
        print(f"❌ Failed to save as NetCDF: {e}")
        raise


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Ocean reconstruction model inference program')
        parser.add_argument('--save_dir', type=str, default='./output_data/recon', help='Path to save inference results')
        parser.add_argument('--src_dir', type=str, default='./src', help='Path to source directory')
        parser.add_argument('--date', type=str, default='20250701',
                            help='Processing date in YYYYMMDD format, default is today')
        parser.add_argument('--surface_file_dir', type=str, default='./input_data/SLA',
                            help='Path to surface file directory')
        parser.add_argument('--sst_file_dir', type=str, default='./input_data/SST',
                            help='Path to SST file directory')
        parser.add_argument('--sss_file_dir', type=str, default='./input_data/SSS',
                            help='Path to SSS file directory')
        parser.add_argument('--deep_file_dir', type=str, default='./input_data/DEEP_LAYER_BACKGROUND',
                            help='Path to deep ocean background data directory')

        args = parser.parse_args()

        ref_date = datetime.strptime(args.date, '%Y%m%d').date()
        date_s = date_series(0, 1, ref_date=ref_date)
        date_s_deep = date_series(7, 1, ref_date=ref_date)

        datatime_str = date_s[-1]
        src_dir = args.src_dir

        # Load surface data (SLA and velocity)
        surface_file_dir = args.surface_file_dir
        file_dict_sla = find_files_by_keyword_and_dates(surface_file_dir, ["allsat"], date_s)
        surface_file = file_dict_sla[date_s[-1]][-1]
        print(f"Surface data loaded successfully: {surface_file}")

        # Load Sea Surface Temperature data
        sst_file_dir = args.sst_file_dir
        file_dict_oisst = find_files_by_keyword_and_dates(sst_file_dir, ["oisst"], date_s)
        sst_file = file_dict_oisst[date_s[-1]][0]
        print(f"SST data loaded successfully: {sst_file}")

        # Load Sea Surface Salinity data
        sss_file = f"{args.sss_file_dir}/model_output_sss_{date_s[-1]}.nc"
        print(f"SSS data loaded successfully: {sss_file}")

        # Load deep ocean background data (7 days before the target date)
        deep_file_dir = args.deep_file_dir
        file_dict_deep = find_files_by_keyword_and_dates(deep_file_dir, ["ocean_vars"], date_s_deep)
        depth_file = file_dict_deep[date_s_deep[-1]][0]
        print(f"Deep background data loaded successfully: {depth_file}")

        # Load normalization statistics
        input_mean = np.load(os.path.join(src_dir, 'Satellite_cmems_mean.npy'))
        input_var = np.load(os.path.join(src_dir, 'Satellite_cmems_std.npy'))

        infer_mean = np.load(os.path.join(src_dir, 'glorys_all_channel_mean.npy'))[1:]
        infer_var = np.load(os.path.join(src_dir, 'glorys_all_channel_std.npy'))[1:]

        # Combine normalization statistics for input and output
        input_mean = np.concatenate([input_mean, infer_mean], axis=0)
        input_var = np.concatenate([input_var, infer_var], axis=0)

        # Load and preprocess input data
        input_data, mask = load_single_day_recon_input(
            surface_file=surface_file,
            sst_file=sst_file,
            sss_file=sss_file,
            depth_file=depth_file,
            is_normalized=True,
            ocean_mean=input_mean,
            ocean_var=input_var
        )

        print(f"Reconstruction model input data processed successfully, data shape: {input_data.shape}")

        # Load the reconstruction model
        model_path = os.path.join(src_dir, "model_onnx", "recon_model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Reconstruction model file not found: {model_path}")

        # Set up ONNX Runtime providers (CUDA if available, otherwise CPU)
        available_providers = onnxruntime.get_available_providers()
        providers = ['CUDAExecutionProvider',
                     'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else [
            'CPUExecutionProvider']

        print(f"Using inference backend: {providers}")
        ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
        ort_inputs = {'input': np.expand_dims(input_data, axis=0)}
        ort_outs = ort_session.run(None, ort_inputs)

        # Post-process model output
        output = ort_outs[2].squeeze(0)
        output = (output * infer_var.reshape(80, 1, 1)) + infer_mean.reshape(80, 1, 1)
        output[np.isnan(mask)] = np.nan

        print(f"Reconstruction inference completed")
        npy2nc(output)
        print(f"Reconstruction results saved successfully")

    except Exception as e:
        print(f"Reconstruction program failed: {e}")
        raise