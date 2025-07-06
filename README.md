# xray-fitting-pipeline
Modular peak-fitting pipeline for SAXS/WAXS data with YAML config and image outputs.

# Batch Peak Fitting Pipeline
This repository provides a Python pipeline for performing batch peak fitting (Gaussian or Pseudo-Voigt) on 2D mesh spectral data stored in HDF5 format. It is optimized for parallel computation using `joblib` and outputs both processed data and diagnostic plots.


## Directory Structure
├── outputs/                 # for generated PNGs and fitted HDF5 file
├── src/                     
│ └── peak_fit_pipeline.py   # Main executable script
├── config.yaml              # Configuration file for datasets and input parameters
├── README.md                # This documentation file
├── requirements.txt         # Python dependencies

---

## Features
- Supports **Gaussian** and **Pseudo-Voigt** peak fitting models
- Parallelized processing for large-scale mesh datasets
- Region-of-interest (ROI) selection and parameter configuration via YAML
- Interactive visualizations with Plotly and Matplotlib
- Diagnosis graphs with single-point fitting
- Outputs:
  - Fitted HDF5 file with parameter maps
  - PNG diagnostic images and visualizations

---

## Installation
Use the provided `requirements.txt` to install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration
Edit the config.yaml file to define:
- Input and output paths
- Fitting model: gaussian or pseudo_voigt
- Aspect ratio of the mesh (e.g., "1001/301")
- Region of interests (ROI) with peak center and half-width from the peak
- Optional: diagnostic single-point fitting mode, and visualizations

---

## Usage
From the root directory, run:

```bash
python src/peak_fit_pipeline.py
```

This will:
- Read the config file
- Load datasets
- Process each defined ROI
- Save results and images in the specified output directory

---

## Output
- *_fit_*.h5: HDF5 file containing fitted parameter maps
- *.png: Heatmap images of fitted parameters

Optional files:
- *_pure.png: Clean (axis-less) images for presentation
- *_diagnostic_point_*.png: Fit and residual plots for se- lected mesh points
- *_azimuthal_avg.html: Interactive average intensity plot

---

## Notes
- Only one HDF5 file is supported per run.
- If the input HDF5 file is opened in another application, the script will abort.
