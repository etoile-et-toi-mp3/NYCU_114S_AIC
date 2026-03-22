# AI Capstone HW1: BEV Projection
**Author:** FU, YONG-WEI (112550107)

## Overview
This .zip file contains the implementation for projecting a Bird's-Eye View (BEV) image onto a front-facing perspective.

## Environment Setup
This script is designed to run seamlessly within the `habitat` conda environment created in **Homework 0**. The required libraries (`numpy` and `opencv-python`) were already installed via the HW0 dependencies.

To activate the environment:
```bash
conda activate habitat
```

## How to Run
Execute the main python script from the root directory:
```bash
python bev2front.py
```

## Usage Instructions
1. Run the script, and the BEV (top-down) image window will appear.
2. Use your mouse to click exactly **four points** on the ground plane in either **clockwise or counterclockwise order** to define a closed region.
3. Press the **`q`** key to close the BEV window and trigger the projection calculation.
4. The script will automatically project the defined region and open a new window displaying the corresponding perspective (front-view) image.
5. Press the **`q`** key again to close the front-view window and exit the program.
6. The resulting projected image is automatically saved as a `.png` file in your directory.