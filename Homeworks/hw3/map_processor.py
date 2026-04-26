import cv2
import numpy as np
from typing import List, Tuple


SCALE_FACTOR = 10000.0 / 255.0 # means that we will be in "m" now
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])
RESOLUTION = 0.05


def load_and_filter_map(point_path: str, color_path: str, height_limit: float = 1.5):
    points = np.load(point_path)
    colors = np.load(color_path)

    # 1. Put all xyz coordinates into the positive quadrant
    min_xyz = np.min(points, axis=0)
    points = points - min_xyz
    
    # 2. scale them from numpy world to real-world mm coordinates
    points = points * SCALE_FACTOR

    # 3. Identify Floor by color
    colors_int = np.round(colors).astype(int)
    y_coords = points[:, 1]
    is_floor = np.all(colors_int == FLOOR_COLOR, axis=1)
    
    # 4. Define vertical limit for the agent
    is_within_height = (y_coords < height_limit) & (y_coords > 0.2)

    # 5. Create walkable_mask and obstacle_mask
    # Walkable is simply the floor
    walkable_mask = is_floor
    
    # Obstacles are points within the height range that are NOT floor and NOT ceiling
    # We include everything from y=0 up to the limit (remaining low points as requested)
    obstacle_mask = (~is_floor) & is_within_height

    # 6. Project metrics to 2D pixel grid indices
    # We use x and z for the 2D plane (x is width, z is height)
    x_coords = points[:, 0]
    z_coords = points[:, 2]

    # Convert meters to pixels based on RESOLUTION (0.05m/px)
    x_idx = np.round(x_coords / RESOLUTION).astype(int)
    z_idx = np.round(z_coords / RESOLUTION).astype(int)

    # 7. Determine Map Dimensions
    # Since we shifted to the positive quadrant, the min index is 0
    w = x_idx.max() + 1
    h = z_idx.max() + 1

    # 8. Initialize maps
    # Background for visual map is white (1.0), representing empty/unknown space
    map_img = np.ones((h, w, 3), dtype=np.float32)
    # Background for occupancy is 0 (free space)
    occupancy_map = np.zeros((h, w), dtype=np.uint8)
    # Background for floor is 0 (no floor detected)
    floor_map = np.zeros((h, w), dtype=np.uint8)

    # 9. Fill the Floor Map
    # Only pixels that have actual "floor" labels in the point cloud
    floor_map[z_idx[walkable_mask], x_idx[walkable_mask]] = 1

    # 10. Fill the Occupancy and Color Map
    # Pixels that have obstacles within the agent's height range
    occupancy_map[z_idx[obstacle_mask], x_idx[obstacle_mask]] = 1
    
    # Fill map_img with colors for visualization (normalized to 0-1)
    obs_colors = colors[obstacle_mask] / 255.0
    map_img[z_idx[obstacle_mask], x_idx[obstacle_mask]] = obs_colors

    # 11. Final Post-Processing: Cleaning the maps
    
    # A. Fill holes in the Floor Map (Morphological Closing)
    # A 5x5 kernel will bridge gaps up to ~0.25m
    kernel_floor = np.ones((5, 5), np.uint8)
    floor_map = cv2.morphologyEx(floor_map, cv2.MORPH_CLOSE, kernel_floor)
    
    # B. Dilation for the Obstacles (Safety Margin)
    kernel_obs = np.ones((4, 4), np.uint8)
    occupancy_map = cv2.dilate(occupancy_map, kernel_obs, iterations=1)

    # 12. Packaging Transformation Parameters
    # We store the min_xyz (the original translation offset), the resolution, 
    # and the scale_factor to allow main.py to map pixels back to the simulator's world.
    transform_params = {
        "min_xyz": min_xyz,
        "resolution": RESOLUTION,
        "scale_factor": SCALE_FACTOR
    }
    
    return map_img, occupancy_map, floor_map, transform_params


def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point.append((x, y))
            print(f"Start selected: ({x}, {y})")

    # Use WINDOW_NORMAL to allow resizing, then force it larger
    cv2.namedWindow("Select Start", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Start", 800, 800)
    cv2.setMouseCallback("Select Start", mouse_callback)
    
    print("Click on the map window to select a start location...")

    while True:
        cv2.imshow("Select Start", (map_img * 255).astype(np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if start_point:
            break
        if key == ord("q"):
            raise RuntimeError("No start selected. Exiting.")

    cv2.destroyWindow("Select Start")
    return start_point[0]


def get_goal_pixels(map_img: np.ndarray, semantic_dict: dict, goal_name: str) -> List[Tuple[int, int]]:
    """function to find all pixels corresponding to the goal object based on color matching."""

    if goal_name.lower() not in semantic_dict:
        raise ValueError(f"Unknown semantic object: {goal_name}. Available options: {list(semantic_dict.keys())}")

    goal_colors = semantic_dict[goal_name.lower()]
    goal_pixels: List[Tuple[float, float]] = []

    for gc in goal_colors:
        gc_norm = np.array(gc) / 255.0
        mask_goal = np.all(np.isclose(map_img, gc_norm, atol=10/255.0), axis=2)
        zs, xs = np.where(mask_goal)
        goal_pixels.extend(list(zip(xs, zs)))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels


def find_navigable_goal(object_pixels: List[Tuple[int, int]], occupancy_map: np.ndarray, floor_map: np.ndarray) -> Tuple[int, int]:
    # 1. Calculate the centroid of the object
    obj_xs = np.array([p[0] for p in object_pixels])
    obj_zs = np.array([p[1] for p in object_pixels])
    centroid_x, centroid_z = np.mean(obj_xs), np.mean(obj_zs)

    # 2. Calculate "Clearance" for every pixel
    # dist_to_obs tells us how many pixels away the nearest obstacle is
    # We invert the occupancy map because distanceTransform measures distance to 0s
    dist_to_obs = cv2.distanceTransform((1 - occupancy_map), cv2.DIST_L2, 5)

    # 3. Identify ROBUST floor candidates
    # We want pixels that are:
    # - On the floor (floor_map == 1)
    # - Not an obstacle (occupancy_map == 0)
    # - Have at least 3-5 pixels of clearance from ANY wall/furniture
    SAFE_DISTANCE = 5 
    safe_floor_mask = (floor_map == 1) & (occupancy_map == 0) & (dist_to_obs >= SAFE_DISTANCE)
    
    floor_zs, floor_xs = np.where(safe_floor_mask)

    # 4. Fallback: If no "safe" floor is found nearby, lower the requirement
    if len(floor_xs) == 0:
        print("Warning: No high-clearance floor found. Lowering safety margin...")
        floor_zs, floor_xs = np.where((floor_map == 1) & (occupancy_map == 0))

    # 5. Find the nearest point among the SAFE candidates
    distances = np.hypot(floor_xs - centroid_x, floor_zs - centroid_z)
    nearest_idx = np.argmin(distances)
    
    return int(floor_xs[nearest_idx]), int(floor_zs[nearest_idx])
