"""
Ocean Forecast Model Inference Script

This script performs inference using trained ocean forecast models to generate
multi-day ocean forecasts from current ocean state observations.

The forecast models take current ocean state data (surface and subsurface) as input
and output predicted ocean fields for lead times of 1-10 days, including
temperature, salinity, velocity, and sea level anomaly.
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

from dataloader import load_single_day_input

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
            match = re.search(r"(\d{8})", filename)
            if match:
                date_str = match.group(1)
                if date_str in date_set and any(k.lower() in filename.lower() for k in keyword_list):
                    full_path = os.path.join(dirpath, filename)
                    result_dict.setdefault(date_str, []).append(full_path)

    return result_dict


def get_depth_day(date_str):
    """
    Calculate the day of year for a given date string.
    
    Args:
        date_str (str): Date string in YYYYMMDD format
        
    Returns:
        int: Day of year (1-366)
    """
    try:
        date = datetime.strptime(date_str, "%Y%m%d")
        day_of_year = date.timetuple().tm_yday
        is_leap_year = (date.year % 4 == 0 and date.year % 100 != 0) or (date.year % 400 == 0)
        if not is_leap_year and day_of_year > 59:
            day_of_year += 1
        return day_of_year
    except Exception as e:
        print(f"❌ Date parsing failed: {e}")
        raise


def npy2nc(npy_data, lead_day):
    """
    Convert numpy array data to NetCDF format and save to file.
    
    Args:
        npy_data (np.ndarray): 3D ocean data array (81 channels, height, width)
        lead_day (int): Forecast lead time in days
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

        variables = ['thetao', 'so', 'uo', 'vo', 'sla']
        layer_index = [list(range(1, 21)), list(range(21, 41)), list(range(41, 61)), list(range(61, 81)), 0]
        array_list = []
        nc_file_single_dir_path = os.path.join(args.save_dir, datatime_str[0:4], datatime_str)
        os.makedirs(nc_file_single_dir_path, exist_ok=True)

        nc_file_path = os.path.join(nc_file_single_dir_path, f"{datatime_str}_lead{lead_day}.nc")
        
        for i in range(len(variables)):
            coords_dict = {
                'time': pd.date_range(datatime_str[:4] + "-" + datatime_str[4:6] + "-" + datatime_str[6:] + " 00:00:00",
                                      periods=1),
                'latitude': lat.astype(np.float32),
                'longitude': lon.astype(np.float32)
            }

            if i < 4:
                dims = ['time', 'depth', 'latitude', 'longitude']
                coords_dict['depth'] = depth.astype(np.float32)
                data = np.expand_dims(npy_data[layer_index[i], :, :], axis=0)
            else:
                dims = ['time', 'latitude', 'longitude']
                data = np.expand_dims(npy_data[layer_index[i], :, :], axis=0)

            nc_data_array = xr.DataArray(data.astype(np.float32), dims=dims, coords=coords_dict)
            array_list.append(nc_data_array)

        ds = xr.Dataset({
            'thetao': array_list[0],
            'so': array_list[1],
            'uo': array_list[2],
            'vo': array_list[3],
            'sla': array_list[4]
        })

        ds.to_netcdf(nc_file_path)
        print(f"✅ Successfully saved forecast results: {nc_file_path}")
    except Exception as e:
        print(f"❌ Failed to save as NetCDF: {e}")
        raise


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='Ocean forecast model inference program')
        parser.add_argument('--save_dir', type=str, default='./output_data/forecast', help='Path to save inference results')
        parser.add_argument('--src_dir', type=str, default='./src', help='Path to source directory')
        parser.add_argument('--date', type=str, default='20250701',
                            help='Processing date in YYYYMMDD format, default is today')
        parser.add_argument('--surface_file_dir', type=str, default='./input_data/SLA',
                            help='Path to surface file directory')
        parser.add_argument('--sst_file_dir', type=str, default='./input_data/SST',
                            help='Path to SST file directory')
        parser.add_argument('--sss_file_dir', type=str, default='./input_data/SSS',
                            help='Path to SSS file directory')
        parser.add_argument('--deep_file_dir', type=str, default='./output_data/recon',
                            help='Path to deep ocean initial field directory (reconstruction results)')

        args = parser.parse_args()

        ref_date = datetime.strptime(args.date, '%Y%m%d').date()
        date_s = date_series(0, 1, ref_date=ref_date)

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

        # Load deep ocean initial field data (from reconstruction results)
        deep_file_dir = f"{args.deep_file_dir}/{datatime_str[0:4]}"
        file_dict_deep = find_files_by_keyword_and_dates(deep_file_dir, ["recon"], date_s)
        depth_file = file_dict_deep[date_s[-1]][0]
        print(f"Deep initial field data loaded successfully: {depth_file}")

        # Load normalization statistics
        input_mean = np.load(os.path.join(src_dir, 'Satellite_cmems_mean.npy'))
        input_var = np.load(os.path.join(src_dir, 'Satellite_cmems_std.npy'))

        infer_mean = np.load(os.path.join(src_dir, 'glorys_all_channel_mean.npy'))
        infer_var = np.load(os.path.join(src_dir, 'glorys_all_channel_std.npy'))

        # Combine normalization statistics for input and output
        input_mean = np.concatenate([input_mean, infer_mean[1:]], axis=0)
        input_var = np.concatenate([input_var, infer_var[1:]], axis=0)

        # Run forecast inference for lead times 1-10 days
        for i in range(1, 11):
            try:
                # Load and preprocess input data
                input_data, mask = load_single_day_input(
                    surface_file=surface_file,
                    sst_file=sst_file,
                    sss_file=sss_file,
                    depth_file=depth_file,
                    is_normalized=True,
                    ocean_mean=input_mean,
                    ocean_var=input_var
                )

                print(f"Forecast day {i} input data processed successfully, data shape: {input_data.shape}")

                # Load the forecast model for the specific lead time
                model_path = os.path.join(src_dir, "model_onnx", f"lead{i}_model.onnx")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")

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
                output = (output * infer_var.reshape(81, 1, 1)) + infer_mean.reshape(81, 1, 1)
                output[np.isnan(mask)] = np.nan

                print(f"✅ Inference completed: lead {i}")
                npy2nc(output, lead_day=i)
                print(f"✅ Inference results saved successfully: lead {i}")

            except Exception as e:
                print(f"❌ Day {i} inference failed: {e}")
                if i == 10:
                    raise
                else:
                    continue

    except Exception as e:
        print(f"Program initialization failed: {e}")
        raise