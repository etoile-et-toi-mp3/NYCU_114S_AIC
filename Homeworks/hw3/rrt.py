import numpy as np
import random
import math
from typing import List, Tuple
import cv2

class Node:
    """Represents a single point in the RRT tree."""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.parent = None

def get_distance_between_nodes(n1: Node, n2: Node) -> float:
    """Calculates the Euclidean distance between two nodes."""
    return math.hypot(n1.x - n2.x, n1.y - n2.y)

def check_collision(n1: Node, n2: Node, occupancy_map: np.ndarray) -> bool:
    """
    Checks if the straight line between two nodes intersects an obstacle.
    Returns True if there is a collision, False if the path is clear.
    """
    x1, y1 = n1.x, n1.y
    x2, y2 = n2.x, n2.y
    dist = get_distance_between_nodes(n1, n2)
    steps = int(dist) # Check pixel-by-pixel along the path
    
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        
        # Check map boundaries
        if y < 0 or y >= occupancy_map.shape[0] or x < 0 or x >= occupancy_map.shape[1]:
            return True
        # Check occupancy (1 = obstacle/inflated boundary)
        if occupancy_map[y, x] == 1:
            return True
            
    return False

def plan_path(start_coords: Tuple[int, int], goal_coords: Tuple[int, int], occupancy_map: np.ndarray, step_size: int = 10, max_iter: int = 20000) -> List[Tuple[int, int]]:
    """
    Executes the RRT algorithm to find a navigable path.
    """
    start_node = Node(start_coords[0], start_coords[1])
    goal_node = Node(goal_coords[0], goal_coords[1])
    
    tree = [start_node]
    height, width = occupancy_map.shape
    
    for _ in range(max_iter):
        # 1. Exploration vs Exploitation: Sample a random point, with a 10% probability to sample the exact goal
        if random.random() < 0.2:
            rnd_node = Node(goal_node.x, goal_node.y)
        else:
            rnd_node = Node(random.randint(0, width - 1), random.randint(0, height - 1))
            
        # 2. Find the nearest existing node in the tree to this random point
        nearest_node = min(tree, key=lambda n: get_distance_between_nodes(n, rnd_node))
        
        # 3. Steer: Calculate a new node one step_size away from the nearest node in the direction of the random point
        theta = math.atan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)
        new_node = Node(
            int(nearest_node.x + step_size * math.cos(theta)),
            int(nearest_node.y + step_size * math.sin(theta))
        )
        new_node.parent = nearest_node
        
        # 4. Collision Check: Verify the line from nearest_node to new_node is free of obstacles
        if not check_collision(nearest_node, new_node, occupancy_map):
            tree.append(new_node)
            
            # 5. Goal Check: If the new node is close enough to the goal, attempt a final connection
            if get_distance_between_nodes(new_node, goal_node) <= step_size:
                if not check_collision(new_node, goal_node, occupancy_map):
                    goal_node.parent = new_node
                    tree.append(goal_node)
                    
                    # Backtrack through parents to build the final route
                    path = []
                    current = goal_node
                    while current is not None:
                        path.append((current.x, current.y))
                        current = current.parent
                        
                    # Reverse the list so it goes from Start -> Goal
                    return path[::-1], tree 
                    
    # Return an empty list if the maximum iterations are reached without finding the goal
    return [], tree

def plan_path_advanced(start: Tuple[int, int], goal: Tuple[int, int], occupancy_map: np.ndarray, max_iter=10000):
    h, w = occupancy_map.shape
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    tree = [start_node]

    # Pre-calculate clearance (Distance to nearest obstacle)
    # 1 - occupancy_map makes obstacles 0 and free space 1
    dist_map = cv2.distanceTransform((1 - occupancy_map).astype(np.uint8), cv2.DIST_L2, 5)

    for i in range(max_iter):
        # 1. Adaptive Goal Bias: Increase probability of picking goal over time
        # Starts at 5% and grows to 25% by the end of iterations
        goal_bias = 0.05 + (0.20 * (i / max_iter))
        
        if random.random() < goal_bias:
            rnd_node = goal_node
        else:
            rnd_node = Node(random.randint(0, w - 1), random.randint(0, h - 1))

        # 2. Find nearest node
        nearest_node = min(tree, key=lambda n: get_distance_between_nodes(n, rnd_node))

        # 3. Variable Step Size: 
        # Base step on local clearance. Min step 3 (precision), Max step 15 (speed).
        clearance = dist_map[nearest_node.y, nearest_node.x]
        dynamic_step = max(3, min(15, int(clearance * 0.8)))

        # Calculate new node position
        theta = math.atan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)
        new_node = Node(
            int(nearest_node.x + dynamic_step * math.cos(theta)),
            int(nearest_node.y + dynamic_step * math.sin(theta))
        )
        new_node.parent = nearest_node

        # 4. Collision check and Tree growth
        if not check_collision(nearest_node, new_node, occupancy_map):
            tree.append(new_node)

            # Check if we can reach the goal
            if get_distance_between_nodes(new_node, goal_node) <= dynamic_step:
                if not check_collision(new_node, goal_node, occupancy_map):
                    goal_node.parent = new_node
                    tree.append(goal_node)
                    
                    # Backtrack to build path
                    path = []
                    curr = goal_node
                    while curr:
                        path.append((curr.x, curr.y))
                        curr = curr.parent
                    return path[::-1], tree

    return [], tree
