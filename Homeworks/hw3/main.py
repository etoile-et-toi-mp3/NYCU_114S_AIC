import sys
import cv2
import numpy as np
from typing import List, Tuple


from map_processor import load_and_filter_map, select_start, get_goal_pixels, find_navigable_goal
from navigator import init_sim, execute_waypoint_path, SENSOR_HEIGHT
from rrt import Node, plan_path


POINT_CLOUD_DATA = "semantic_3d_pointcloud/point.npy"
COLOR_DATA = "semantic_3d_pointcloud/color0255.npy"

# Sample semantic color and index dictionaries for a few object categories. 
# Check hw0/replica_v1/apartment_0/habitat/info_semantic.json and 
# hw3/color_coding_semantic_segmentation_classes.xlsx for the full list of 
# categories and their corresponding colors and indices.
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cushion": [[255, 9, 92]],
        "sofa": [[10, 0, 255]],
        "stair": [[173, 255, 0]],
        "cooktop": [[7, 255, 224]],
    },
    "indices": {
        "rack": 8,
        "cushion": 430,
        "sofa": 196,
        "stair": 192,
        "cooktop": 280,
    },
}


def pick_goal(map_img, occupancy_map, floor_map) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination ('rack', 'cooktop', 'sofa', 'cushion', 'stair'): "
    goal_prompt = input(prompt).strip().lower()
    
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    # 1. Get the raw semantic pixels
    object_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    
    # 2. Use the floor-aware logic to find a valid navigation point
    goal = find_navigable_goal(object_pixels, occupancy_map, floor_map)
    
    return goal_prompt, goal


def visualize_path(map_img: np.ndarray, path: List[Tuple[int, int]], tree: List[Node], start: Tuple[int, int], goal: Tuple[int, int]):
    """Draws the RRT tree and final path over the semantic map."""
    # Convert float map to standard BGR image for drawing
    vis_img = (map_img.copy() * 255).astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Draw the RRT exploration tree
    for node in tree:
        if node.parent is not None:
            # Gray branches
            cv2.line(vis_img, (node.parent.x, node.parent.y), (node.x, node.y), (150, 150, 150), 1)
        # Purple nodes
        cv2.circle(vis_img, (node.x, node.y), 1, (200, 0, 200), -1) 

    # Draw the final path waypoints
    if path:
        for i in range(len(path) - 1):
            # Red path
            cv2.line(vis_img, path[i], path[i+1], (0, 0, 255), 2)

    # Highlight Start and Goal
    cv2.circle(vis_img, start, 2, (0, 255, 0), -1) # Green start
    cv2.circle(vis_img, goal, 2, (255, 0, 0), -1)  # Blue goal

    # Create a resizable window and scale it up to 800x800
    cv2.namedWindow("RRT Path Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RRT Path Output", 800, 800)

    cv2.imshow("RRT Path Output", vis_img)
    print("Press any key on the map window to continue to the simulation...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_in_sim(start_world, world_path, goal_prompt):
    """
    Initializes the simulator and executes the calculated path.
    """
    start_x, start_z = start_world
    
    # Initialize the Habitat simulator at the translated world start position
    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)
    
    # Get the semantic index from the hardcoded dictionary for highlighting
    goal_idx = SEMANTIC_DICTS["indices"][goal_prompt]
    
    print(f"Executing movement toward {goal_prompt} (Index: {goal_idx})...")
    execute_waypoint_path(world_path, sim, agent, goal_idx)


def main():
    """Entry point."""
    print("=== Step 1: Processing the 3D Map ===")
    map_img, occupancy_map, floor_map, transform_params = load_and_filter_map(
        POINT_CLOUD_DATA, 
        COLOR_DATA,
        height_limit=SENSOR_HEIGHT
    )

    # print("=== Debug: Visualizing Maps ===")
    # floor_vis = (floor_map * 255).astype(np.uint8)
    # occ_vis = (occupancy_map * 255).astype(np.uint8)

    # # Create windows
    # cv2.namedWindow("Floor Map (White = Walkable)", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Floor Map (White = Walkable)", 600, 600)
    # cv2.imshow("Floor Map (White = Walkable)", floor_vis)

    # cv2.namedWindow("Occupancy Map (White = Obstacle)", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Occupancy Map (White = Obstacle)", 600, 600)
    # cv2.imshow("Occupancy Map (White = Obstacle)", occ_vis)

    # print("Review the maps. Press any key to continue to Start Selection...")
    # cv2.waitKey(0)
    # cv2.destroyWindow("Floor Map (White = Walkable)")
    # cv2.destroyWindow("Occupancy Map (White = Obstacle)")

    print("\n=== Step 2: Selecting Agent Start and Goal Positions ===")
    goal_prompt, goal = pick_goal(map_img, occupancy_map, floor_map)
    start = select_start(map_img)
    print(f"Goal pixel selected at coordinates: {goal}")

    print("\n=== Step 3: Executing Path Planning (RRT) ===")
    print("Planning path, this may take a moment...")
    path, tree = plan_path(start, goal, occupancy_map)
    
    if not path:
        print("Planner could not find a path. Visualizing the tree to see where it got stuck...")
        visualize_path(map_img, path, tree, start, goal)
        sys.exit(1)

    print("\n=== Step 4: Visualizing the Planned Path ===")
    visualize_path(map_img, path, tree, start, goal)

    print("\n=== Step 5: Translating Path to Habitat Simulator ===")
    world_path = []
    min_xyz = transform_params["min_xyz"]
    res = transform_params["resolution"]
    sf = transform_params["scale_factor"]

    for px, pz in path:
        # 1. (px * res) is the distance from the start of the map in meters.
        # 2. (min_xyz * sf) is the original world minimum coordinate scaled to meters.
        # We DO NOT divide by sf here; we use the scaled meters directly.
        world_x = (px * res) + (min_xyz[0] * sf)
        world_z = (pz * res) + (min_xyz[2] * sf)
        
        world_path.append((world_x, world_z))

    print(f"Translated {len(world_path)} waypoints to Habitat coordinates.")
    run_in_sim(world_path[0], world_path, goal_prompt)


if __name__ == "__main__":
    main()