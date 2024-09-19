# SLM-based Pupil Segmentation for LUT Correction and Wavefront Sensing

This project provides Python toolkits for controlling Spatial Light Modulators (SLMs) and synchronized camera acquisition.

It enables both wavefront sensing and Look-Up Table (LUT)-based corrections through pupil segmentation techniques.

## Core Modules

### 1. **Gratings** (`gratings.py`)

This module generates and manipulates various gratings and masks on a 2D canvas.

Key Features:

- Generate blazed and binary gratings
- Position gratings accurately and apply circular or rectangular masks
- Create rotational coordinates for complex arrangements

### 2. **HPC_SLM** (`hpc_slm.py`)

This module controls Hamamatsu Photonics SLMs through two main classes:

- **`HPC_SLM`**: Manages window display and transfers gray-scale images (`np.ndarray`) to the SLM via DVI..
- **`HPC_SLM_Calib`**: Extends the base class, adding calibration capabilities using local or global LUTs to convert phase data into gray values.

Key Features:

- SLM initialization and window management
- Calibration data loading and application
- Phase-to-gray value conversion with correction options

### 3. **GxCam** (`MyGxCam.py`)

This module provides an interface for controlling Daheng USB Vision3 cameras.

Key Features:

- Camera initialization and configuration
- Trigger mode control and exposure time adjustment
- Image acquisition (single and multiple frames)

## Demonstration Notebooks

Two Jupyter notebooks are included, demonstrating pupil segmentation techniques in different scenarios.

### 1. **Pupil Segmentation FullyRot - LUT** (`pupilseg_lut.ipynb`)

Demonstrates a 'FullyRot' pattern for local and non-linear LUT correction, where a mask rotates while applying the LUT.

Key Features:

- Utilizes a binary grating on the SLM
- Implements LUT adjustments to the rotating mask

### 2. **Pupil Segmentation CenterFix - Phase Shift** (`pupilseg_ps.ipynb`)

Demonstrates a 'CenterFix' pattern for wavefront sensing, where a mask is fixed in the center, and another rotates while applying phase shifts.

Key Features:

- Uses a full blazed grating on the SLM
- Applies phase shifts to the rotating mask

> These interactive notebooks make parameter adjustment and data visualization straightforward during experiments.

## Usage

1. Install all required dependencies (see the [Requirements](#requirements) section).
2. Open the desired Jupyter notebook (`pupilseg_lut.ipynb` or `pupilseg_ps.ipynb`).
3. Adjust parameters according to your setup.
4. Execute the cells to run the pupil segmentation experiments.

## Requirements

- Python 3.12
- Jupyter Notebook
- Required libraries: `numpy`, `matplotlib`, `scipy`, `tqdm`
  
  > Note: For the Daheng camera interface, two versions of the `gxipy` library are available:
  > 1. The official version provided by Daheng (included in this project).
  > 2. An unofficial version on PyPI: `iai-gxipy` (use with caution).

## Data Output

Acquired images and experiment parameters are saved in MATLAB (.mat) files with timestamped filenames.

## Notes

This project is tailored to work with specific hardware, namely Daheng cameras and Hamamatsu Photonics SLMs.

If you are using different camera or SLM models, you may need to modify the respective modules (`MyGxCam.py` for cameras and `hpc_slm.py` for SLMs) or adjust the relevant sections in the Jupyter notebooks for compatibility with your hardware.
