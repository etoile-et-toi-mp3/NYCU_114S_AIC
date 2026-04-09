# HW2: 3D Scene Reconstruction
**Author:** FU, YONG-WEI  
**Student ID:** 112550107

## Environment Setup
This script is designed to run seamlessly within the `habitat` conda environment created in **Homework 0**. The required libraries (`numpy` and `opencv-python`) were already installed via the HW0 dependencies.

To activate the environment:
```bash
conda activate habitat
```

## Usage

### Step 1: Data Collection
Before reconstruction, you must collect the RGB-D data from the Replica apartment scene using `load.py`.
```bash
# Collect data for Floor 1
python load.py -f 1

# Collect data for Floor 2
python load.py -f 2
```

### Step 2: 3D Reconstruction
The `reconstruct.py` script supports both the standard Open3D registration pipeline and a custom Point-to-Plane ICP implementation.

**To run using Open3D's ICP:**
```bash
python reconstruct.py -f 1 -v open3d
```

**To run using Custom ICP (Bonus Task):**
```bash
python reconstruct.py -f 1 -v my_icp
```

### Argument Flags:
- `-f, --floor`: Choose between floor `1` or `2`.
- `-v, --version`: Choose the registration method (`open3d` or `my_icp`).
