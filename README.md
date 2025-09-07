# KanHai (SkyOcean)



KanHai is a novel deep learning framework for real-time, satellite-driven ocean state forecasting in the Northwestern Pacific at 1/8° resolution, covering depths from 0 to 650 m. It employs a two-stage reconstruction-prediction approach to predict SLA, temperature, salinity, and velocity components.

## Installation

### Dependencies

- Python 3.11
- ONNX Runtime
- xarray

create conda environment

```bash
conda create --name kanhai python=3.11 
```

```bash
conda activate kanhai
```

Install dependencies:

```bash
pip install onnxruntime-gpu
```

```bash
conda install xarray dask netCDF4 bottleneck
```

Clone the repository:

```bash
git clone https://github.com/skyocean-kanhai/KanHai.git
cd kanhai
```

## Quick Start

### 1. Ocean Reconstruction

Reconstruct 3D ocean fields from surface observations:

```bash
python inference_recon.py --date 20250701
```

### 2. Ocean Forecasting

Generate 10-day ocean forecasts:

```bash
python inference_forecast.py --date 20250701
```

## Project Structure

The KanHai project follows a clean, organized structure:

```
kanhai_release/
├── input_data/                          # Input data directory
│   ├── SLA/                            # Sea Level Anomaly data
│   │   └── 2025/                       # Year-based organization
│   ├── SST/                            # Sea Surface Temperature data
│   ├── SSS/                            # Sea Surface Salinity data
│   └── DEEP_LAYER_BACKGROUND/          # Background subsurface data
├── output_data/                        # Output results directory
│   ├── recon/                          # Reconstruction results
│   │   └── 2025/                       # Year-based organization
│   └── forecast/                       # Forecast results
│       └── 2025/                       # Year-based organization
├── src/                                # Source code and models
│   ├── model_onnx/                     # KanHai ONNX models
│   │   ├── recon_model.onnx           # Reconstruction model
│   │   ├── lead1_model.onnx           # 1-day forecast model
│   │   ├── lead2_model.onnx           # 2-day forecast model
│   │   ├── ...                        # Additional forecast models
│   │   └── lead10_model.onnx          # 10-day forecast model
│   ├── template.nc                     # NetCDF template
│   ├── Satellite_cmems_mean.npy       # Satellite data normalization
│   ├── Satellite_cmems_std.npy        # Satellite data normalization
│   ├── glorys_all_channel_mean.npy    # GLORYS data normalization
│   └── glorys_all_channel_std.npy     # GLORYS data normalization
├── dataloader.py                       # Data loading utilities
├── inference_recon.py                  # Reconstruction inference script
├── inference_forecast.py               # Forecasting inference script
└── README.md                           # This file
```

## Usage Examples

### Command Line Interface

#### Reconstruction

```bash
python inference_recon.py \
    --date 20250701 \
    --save_dir ./output_data/recon \
    --surface_file_dir ./input_data/SLA \
    --sst_file_dir ./input_data/SST \
    --sss_file_dir ./input_data/SSS \
    --deep_file_dir ./input_data/DEEP_LAYER_BACKGROUND
```

#### Forecasting

```bash
python inference_forecast.py \
    --date 20250701 \
    --save_dir ./output_data/forecast \
    --surface_file_dir ./input_data/SLA \
    --sst_file_dir ./input_data/SST \
    --sss_file_dir ./input_data/SSS \
    --deep_file_dir ./output_data/recon
```

## Output Format

### NetCDF Files

The KanHai models output NetCDF files with the following structure:

**Reconstruction Output** (`output_data/recon/YYYY/recon_YYYYMMDD.nc`):

- `thetao`: Temperature at 20 depth levels
- `so`: Salinity at 20 depth levels  
- `uo`: Zonal velocity at 20 depth levels
- `vo`: Meridional velocity at 20 depth levels

**Forecast Output** (`output_data/forecast/YYYY/YYYYMMDD/YYYYMMDD_leadN.nc`):

- `thetao`: Temperature at 20 depth levels
- `so`: Salinity at 20 depth levels
- `uo`: Zonal velocity at 20 depth levels  
- `vo`: Meridional velocity at 20 depth levels
- `sla`: Sea Level Anomaly

### Coordinate System

- **Longitude**: 100°E to 159.875°E (0.125° resolution)
- **Latitude**: 0°N to 49.875°N (0.125° resolution)
- **Depth**: 20 selected levels, including depths of 0.49 m, 2.65 m, 5.08 m, 7.93 m, 11.41 m, 15.81 m, 21.60 m, 29.44 m, 40.34 m, 55.76 m, 77.85 m, 92.32 m, 109.73 m, 130.67 m, 155.85 m, 186.13 m, 222.48 m, 318.13 m, 453.94 m, and 643.57 m, corresponding to GLORYS12 indices [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32].
- **Time**: Single time step for each forecast/reconstruction

## Data Sources

The KanHai models are trained on high-quality oceanographic data:

- **Surface Data**: CMEMS satellite products (SLA, SST, SSS)
- **Subsurface Data**: GLORYS ocean reanalysis
- **Validation**: Independent ocean observations and reanalysis products
- **Coverage**: Global ocean with focus on regional accuracy

## Model Files

The KanHai system includes the following trained models:

- `recon_model.onnx`: Ocean reconstruction model (85 → 80 channels)
- `lead1_model.onnx` to `lead10_model.onnx`: Forecast models for 1-10 day lead times (85 → 81 channels)

## Citation

If you use KanHai (SkyOcean) in your research, please cite:

```bibtex
@software{kanhai_skyocean,
  title={KanHai (SkyOcean): Deep Learning for Ocean Reconstruction and Forecasting},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/kanhai-skyocean}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions to KanHai (SkyOcean) are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support regarding KanHai (SkyOcean), please open an issue on GitHub.

## Acknowledgments

- CMEMS for providing satellite ocean data
- GLORYS team for ocean reanalysis products
- ONNX Runtime team for efficient model inference
- The oceanographic community for data validation and feedback

---

**KanHai (SkyOcean)** - Bridging the gap between satellite observations and deep ocean understanding through advanced AI.
