import os
import re
import glob
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time
import cv2
from pathlib import Path

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model.
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
DEPTH_SCALE = 1000.0 #

def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    TASK 1: Geometric Unprojection
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).
    """
    # 1. Convert depth back to real-world meters using DEPTH_SCALE = 1000.0
    z_matrix = depth_image.astype(np.float32) / DEPTH_SCALE
    
    # 2. Create a boolean mask to filter out invalid depth pixels (depth == 0)
    valid_mask = z_matrix > 0

    # 3. Create a coordinate grid for (u, v) pixels using np.meshgrid
    h, w = depth_image.shape
    
    # np.meshgrid takes two 1D arrays and creates two 2D grid matrices.
    # If this was a tiny 3x3 image, 'u' (X-coordinates/columns) would look like:
    # [[0, 1, 2],
    #  [0, 1, 2],
    #  [0, 1, 2]]
    # And 'v' (Y-coordinates/rows) would look like:
    # [[0, 0, 0],
    #  [1, 1, 1],
    #  [2, 2, 2]]
    # Using them as look up tables, we can get the (u, v) coordinates for every pixel, and
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 4. Apply the mask to extract only valid coordinates
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z_matrix[valid_mask]
    
    # 5. Apply the Pinhole Camera Model formulas to get the 3D coordinates (in the camera frame)
    x = (u_valid - CX) * z_valid / FX
    y = -(v_valid - CY) * z_valid / FY # Y in the computer is the inverse of the real-world Y axis
    z = -z_valid # Camera looks towards -Z axis
    
    # 6. Stack into an (N, 3) array for Open3D
    points_3d = np.vstack((x, y, z)).T
    
    # 7. Extract corresponding RGB colors and normalize to [0, 1] range
    colors_norm = rgb_image[valid_mask] / 255.0
    
    # 8. Assign to Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    """
    Pre-processing: Voxelization and Normal Estimation
    """
    # 1. Voxel Downsampling: setup a lot of small cubes in the 3D space, and average them together (both XYZ and color) into one single point per grid cell to downsample the data.
    # the bigger the voxel_size is, the less data it retains.
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    
    # 2. Now we got less points. For each point, we do the same thing:
        # We find all the points in its small neighborhood (defined by radius_normal) and fit a plane to these points. (we do this with PCA)
        # Record "the normal vector of the plane" to be "the normal of THAT point".
    radius_normal = voxel_size * 2.0
    pcd_downsampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    # 3. Now we got less points and their normals. Fast Point Feature Histograms (FPFH) is an algorithm that for each point:
        # Look at its neighborhood's (defined by radius_feature) coordinates and normals
        # Compute a 33-dimensional embedding that describes the local geometry around that point. (This point belong to a wall? A corner? An edge? A plane? etc.)
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_downsampled,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    return pcd_downsampled, pcd_fpfh

def local_icp_algorithm(downsampled_source, downsampled_target, guessed_transform, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED) 
    """
    # ICP (Iterative Closest Point) is an algorithm that refines the RANSAC result
    # it first transforms the source pcd with the RANSAC result.
    # Since we are using Point-to-Plane ICP, we want to minimize the distance from
    # each transformed source point to the normal plane of its nearest neighbor in the target pcd.
    # Mathematically, minimizing \sum{(d_i . n_i)^2} where d_i is the difference vector from the neighbor to the transformed point, and n_i is the normal of that neighbor.
    
    # We add ConvergenceCriteria to prevent ICP from running too many iterations
    # relative_fitness: stop if the overlap improvement is less than 0.000001
    # relative_rmse: stop if the error improvement is less than 0.000001
    # max_iteration: stop after 30 steps (ICP usually converges very fast anyway)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=30 
    )
    
    result = o3d.pipelines.registration.registration_icp(
        downsampled_source, 
        downsampled_target, 
        threshold, 
        guessed_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    return result

def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform, max_iterations=20):
    """
    Optimized Custom ICP: High speed, slightly lower precision.
    """
    T_global = initial_transform.copy()
    
    # 1. DOWN-SAMPLE AGAIN (The "Speed Secret")
    # Even if the main loop is 0.04, we sample even fewer points for the math.
    # 2000 points is plenty for indoor scenes.
    source_pcd_small = source_pcd.random_down_sample(sampling_ratio=0.1) 
    
    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    target_points = np.asarray(target_pcd.points)
    target_normals = np.asarray(target_pcd.normals)
    
    for i in range(max_iterations):
        source_temp = deepcopy(source_pcd_small)
        source_temp.transform(T_global)
        curr_points = np.asarray(source_temp.points)
        
        # Lists to store vectorized data
        A_list = []
        b_list = []

        # We still need to find neighbors. 
        # This loop is faster because len(curr_points) is now small.
        for p in curr_points:
            [k, idx, dist_sq] = target_tree.search_knn_vector_3d(p, 1)
            
            if k > 0 and dist_sq[0] < 0.0025: # 0.05m threshold
                q = target_points[idx[0]]
                n = target_normals[idx[0]]
                
                # Cross product p x n
                c = np.cross(p, n)
                A_list.append(np.hstack((c, n)))
                b_list.append(np.dot(q - p, n))

        if len(A_list) < 10: break
        
        # 2. VECTORIZED SOLVE (Massive Speedup)
        A = np.array(A_list)
        b = np.array(b_list)
        
        # Build AtA and Atb in two lines
        AtA = A.T @ A
        Atb = A.T @ b
        
        try:
            x = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError: break

        # 3. FAST UPDATE
        dR = o3d.geometry.get_rotation_matrix_from_xyz(x[:3])
        T_delta = np.identity(4)
        T_delta[:3, :3] = dR
        T_delta[:3, 3] = x[3:]
        
        T_global = T_delta @ T_global
        
        # 4. LOOSE TOLERANCE
        if np.linalg.norm(x) < 1e-3: # Stop if the "nudge" is tiny
            break

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_global
    return result

def reconstruct(args):
    # hardcode variables
    rgb_dir = Path(args.data_root) / "rgb"
    rgb_files = sorted(rgb_dir.glob("*.png"), key=lambda x: int(x.stem))
    rgb_files = [str(p) for p in rgb_files]
    depth_dir = Path(args.data_root) / "depth"
    depth_files = sorted(depth_dir.glob("*.png"), key=lambda x: int(x.stem))
    depth_files = [str(p) for p in depth_files]

    voxel_size = 0.04 # 4cm
    ransac_threshold = voxel_size * 2.5 # 0.1 meters: RANSAC needs a wider net to catch fast rotations
    icp_threshold = voxel_size * 1.5 # 0.06 meters: ICP needs enough room to catch height changes on the stairs

    # Load Ground Truth Poses
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
    else:
        print(f"Warning: Ground Truth pose file not found at {gt_pose_path}. GT data and poses will be empty.")
        gt_data = []

    gt_poses = []
    for p in gt_data:
        # we create the matrix for the transformation of:
        # [ x' ]   [ R_11 R_12 R_13 t_x ] [ x ]
        # [ y' ] = [ R_21 R_22 R_23 t_y ] [ y ]
        # [ z' ]   [ R_31 R_32 R_33 t_z ] [ z ]
        # [ 1  ]   [ 0    0    0    1   ] [ 1 ]
        mat = np.identity(4)
        mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix() # turn quaternion numbers into a rotation matrix
        mat[:3, 3] = [p[0], p[1], p[2]] # translation part
        gt_poses.append(mat)
        # the gt_data and these poses are all in the simulator's coordinate space
    gt_poses = np.stack(gt_poses) if len(gt_poses) > 0 else np.array([]) # shape (N, 4, 4)

    # Initialize the array for our guessed camera poses (also in the simulator's coordinate space) and the accumulated point cloud
    guessed_camera_poses = [gt_poses[0].copy()] if len(gt_poses) > 0 else [np.identity(4)] # yes we leak the GT pose for the first frame, as we need a starting point for the trajectory
    accumulated_pcd = o3d.geometry.PointCloud()

    # Load Frame 0
    rgb_0 = cv2.cvtColor(cv2.imread(rgb_files[0]), cv2.COLOR_BGR2RGB)
    depth_0 = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
    pcd_0 = depth_image_to_point_cloud(rgb_0, depth_0) # this pcd is in the camera's local coordinate space now
    downsampled_0, fpfh_0 = preprocess_point_cloud(pcd_0, voxel_size)
    
    # Transform Frame 0 into the Global Coordinate space before accumulating
    downsampled_0_in_global = deepcopy(downsampled_0)
    downsampled_0_in_global.transform(guessed_camera_poses[0]) # Multiplying (x' = mat * x) transforms the point cloud from the camera's local space to the global space
    accumulated_pcd += downsampled_0_in_global
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size)

    downsampled_prev = downsampled_0
    fpfh_prev = fpfh_0
    # Reconstruction Loop
    for i in range(1, len(rgb_files)):
        print(f"Processing Frame {i}/{len(rgb_files)-1}...")
        
        rgb_i = cv2.cvtColor(cv2.imread(rgb_files[i]), cv2.COLOR_BGR2RGB)
        depth_i = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        pcd_i = depth_image_to_point_cloud(rgb_i, depth_i)
        downsampled_i, fpfh_i = preprocess_point_cloud(pcd_i, voxel_size)
        
        # Use RANSAC (RANdom SAmple Consensus) to calculate a rough transformation between the current frame and the previous frame based on their FPFH embeddings.
        # It first randomly samples 3 points from the prev frame and the current frame that has similar FPFH embeddings,
        # calculates how to move from the 3 prev points to the 3 current points (definitely doable in 3D space),
        # and then checks how many other points in the room also "match well" with this same transformation.
        # ("Match well": If a transformed point and its nearest neighbor in the new frame is within distance_threshold, this counts as a "vote" (Consensus!).
        # The points won't be too dense since we voxel downsampled them.
        # It repeats this process for 1,000,000 iterations or until it finds a transformation that has 99% inliers, and returns the best transformation it found.
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            downsampled_i, downsampled_prev, fpfh_i, fpfh_prev, True,
            ransac_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), # <--- THE TRUE FIX
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1500000, 0.99)
        )
        
        # Use ICP to better "refine" the RANSAC transformation.
        # Note that RANSAC is like "taking huge steps, but might be inaccurate", while ICP is like "taking small steps, but more precise".
        # The icp_result is a transformation also, but should be more accurate then the RANSAC result.
        if args.version == 'open3d':
            icp_result = local_icp_algorithm(downsampled_i, downsampled_prev, ransac_result.transformation, icp_threshold)
        else:
            icp_result = my_local_icp_algorithm(downsampled_i, downsampled_prev, ransac_result.transformation)
        
        # Update guessed_camera_poses and accumulate points
        # New guessed camera pose = the latest guessed camera pose * the better transformation from ICP
        new_guessed_camera_pose = np.dot(guessed_camera_poses[-1], icp_result.transformation)
        guessed_camera_poses.append(new_guessed_camera_pose)
        
        # Transform the current frame's downsampled pcd into the global coordinate, and add it to the accumulated point cloud.
        downsampled_i_in_global = deepcopy(downsampled_i)
        downsampled_i_in_global.transform(new_guessed_camera_pose)
        accumulated_pcd += downsampled_i_in_global
        accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size) # use voxel again to prevent the point cloud from exploding in size
        
        # prepare for the next loop
        downsampled_prev = downsampled_i
        fpfh_prev = fpfh_i
    
    print(f"Reconstruction complete. Total points: {len(accumulated_pcd.points)}")
    return accumulated_pcd, guessed_camera_poses, gt_poses

def visualize_and_evaluate(accumulated_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization
    """
    # --- Part 1: Post-processing (Removing Ceiling) ---
    # Moving this here keeps the reconstruction function focused on geometry
    print(f"Post-processing: Removing Floor {args.floor} ceiling for better visibility...")
    points = np.asarray(accumulated_pcd.points)
    colors = np.asarray(accumulated_pcd.colors)
    
    # Adjusting ceiling height based on the floor
    if args.floor == 1:
        ceiling_height = 0.6  # Chops off the Floor 1 roof
    else:
        ceiling_height = 2.6  # Chops off the Floor 2 roof (since it starts higher)

    mask = points[:, 1] < ceiling_height
    
    # Apply the mask to filter out the high Y-values (the roof)
    accumulated_pcd.points = o3d.utility.Vector3dVector(points[mask])
    accumulated_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    # --- Part 2: Evaluation (L2 Distance) ---
    pred_xyz = np.array([pose[:3, 3] for pose in predicted_cam_poses]) # yes that syntax only takes the t_x, t_y, t_z
    gt_xyz = np.array([pose[:3, 3] for pose in gt_poses])
    
    # Align lengths to avoid index errors if one list is shorter
    print(f"Predicted trajectory length: {len(pred_xyz)}, Ground Truth trajectory length: {len(gt_xyz)}")

    # Calculate Mean L2 Distance 
    distances = np.linalg.norm(pred_xyz - gt_xyz, axis=1) # a list of the length of the error vector for each frame
    mean_l2_error = np.mean(distances)
    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    # --- Part 3: Create LineSets for Trajectories ---
    lines = [[i, i + 1] for i in range(len(pred_xyz) - 1)]
    
    # Linesets (Keep these for the 'path' look)
    pred_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pred_xyz),
        lines=o3d.utility.Vector2iVector(lines)
    )
    pred_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
    
    def create_sphere_trajectory(xyz_list, color):
        """Helper to create a collection of spheres for poses"""
        spheres = o3d.geometry.TriangleMesh()
        for pt in xyz_list[::2]: # [::2] takes every 2nd frame so it's not too crowded
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03) # Adjust size here
            sphere.paint_uniform_color(color)
            sphere.translate(pt)
            spheres += sphere
        return spheres

    # Create Spheres (Red for Pred, Black for GT)
    pred_spheres = create_sphere_trajectory(pred_xyz, [1, 0, 0])
    gt_spheres = create_sphere_trajectory(gt_xyz, [0, 0, 0])

    # --- Part 4: Visualization ---
    # We add the spheres to the list
    o3d.visualization.draw_geometries(
        [accumulated_pcd, pred_lineset, pred_spheres, gt_spheres], 
        window_name="Improved Trajectory View"
    )
    
    return mean_l2_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)
    print(f"Total execution time: {time.time() - start_time:.2f}s")
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)
