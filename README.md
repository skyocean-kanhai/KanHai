# KanHai (SkyOcean)

![flowchart](https://github.com/skyocean-kanhai/KanHai/blob/main/figs/flowchart.jpg)

![arch](https://github.com/skyocean-kanhai/KanHai/blob/main/figs/model_arch.png)

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

Please download the complete project files from the cloud drive, including input examples and model weights.

```bash
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

This section describes the preparation of input data for the model. The area must include:

- Longitude: 100°E to 159.875°E (0.125° resolution, 480 grid points)

- Latitude: 0°N to 49.875°N (0.125° resolution, 400 grid points)

The resolution of all data must be interpolated to 1/8°.The Kanhai model is driven by the following data:

- **L4 Sea Level Anomaly (SLA) and Geostrophic Currents (u, v)**: From [Copernicus Marine Service product SEALEVEL_GLO_PHY_L4_NRT_008_046](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description) (0.125° resolution, daily).
- **L4 Sea Surface Temperature (SST)**: [NOAA OISST v2.1 AVHRR dataset](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/) (0.25° resolution, daily). Requires resampling to 0.125°.
- **Sea Surface Salinity (SSS)**: For real-time inference, we integrate data from two L3 products, [SMOS SSS](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_SSS_L3_MYNRT_015_014/description) and [SMAP SSS](https://podaac.jpl.nasa.gov/dataset/SMAP_JPL_L2B_SSS_CAP_V5#), to enhance accuracy and timeliness. Alternatively, users can opt for direct inference using the [Copernicus L4 product MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013/description) (0.125° resolution, daily), which ensures greater consistency.
- **Background Fields**: From [Copernicus GLOBAL_ANALYSISFORECAST_PHY_001_024](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description) (0.083° resolution, daily). Includes:
  - Currents (daily): cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m (eastward/northward sea water velocity).
  - Salinity (daily): cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m (sea water salinity).
  - Temperature (daily): cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m (sea water potential temperature).
    Requires resampling to 0.125°.

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

We gratefully acknowledge the following institutions for providing essential datasets:

- [Copernicus Marine Service](https://data.marine.copernicus.eu/) for the L4 Sea Level Anomaly (SLA), Geostrophic Currents (u, v), Sea Surface Salinity (SSS) L4, and Background Fields.
- [NOAA](https://www.ncei.noaa.gov/) for the L4 Sea Surface Temperature (SST) OISST v2.1 AVHRR dataset.
- [ESA](https://smos-diss.eo.esa.int/) and [NASA](https://podaac.jpl.nasa.gov/) for the SMOS and SMAP SSS L3 products, respectively.CMEMS for providing satellite ocean data

We would like to thank the following GitHub repositories:

- **[ConvIR](https://github.com/c-yn/ConvIR)**: An innovative convolutional network for image restoration, providing open-source code and pre-trained models for tasks like dehazing, desnowing, deraining, and deblurring.
- **[ChaosBench](https://github.com/leap-stc/ChaosBench)**: A multi-channel, physics-based benchmark for subseasonal-to-seasonal climate prediction, offering ERA5/LRA5/ORAS5 data, baselines, and differentiable physics metrics.

---

**KanHai (SkyOcean)** - Bridging the gap between satellite observations and deep ocean understanding through advanced AI.
