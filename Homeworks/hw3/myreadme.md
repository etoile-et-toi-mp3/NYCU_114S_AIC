# HW3: Robot Navigation Framework
Author: FU, YONG-WEI  
Student ID: 112550107

## Environment Setup
This project runs within the `habitat` conda environment established in Homework 0. To activate the environment:
```bash
conda activate habitat
```

## Usage

### Running the Navigation Pipeline
The entire process—from processing the 3D point cloud to executing the path in the simulator—is managed by `main.py`.

```bash
python main.py
```

### Execution Steps:
1.  **Map Processing:** The script loads the 3D point cloud (`point.npy`) and color data (`color0255.npy`), filters them, and generates a 2D occupancy grid.
2.  **Goal Selection:** You will be prompted in the terminal to enter a semantic destination (e.g., `rack`, `cooktop`, `sofa`, `cushion`, `stair`). 
3.  **Start Selection:** A window will appear showing the processed map. **Left-click** on a white (walkable) area to set the agent's starting position.
4.  **RRT Planning:** The algorithm will compute a collision-free path. A visualization window will appear showing the RRT exploration tree (purple) and the final path (red). Press any key on this window to proceed.
5.  **Habitat Simulation:** The Habitat simulator will launch. You can watch the agent navigate to the target in the RGB, Depth, and Semantic windows. The target object will be highlighted with a red mask.

## Src Files
- `main.py`: The entry point that orchestrates the perception, planning, and simulation steps.
- `map_processor.py`: Contains logic for 3D-to-2D projection, coordinate transformation, and safe goal point selection.
- `rrt.py`: Implementation of the Rapidly-exploring Random Tree algorithm with goal biasing and collision checking.
- `navigator.py`: Handles the Habitat-Sim initialization, discrete movement control ($0.02m$ steps), and semantic visual feedback.

## Key Parameters
- **Resolution:** 0.05m per pixel.
- **RRT Step Size:** 10 pixels.
- **Goal Bias:** 20% (probability to sample the goal directly).
- **Movement Step:** 0.02m (tuned for high-precision navigation).