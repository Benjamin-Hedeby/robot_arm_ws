# cspace_grid.py
import numpy as np

class CSpaceGrid:
    def __init__(self, joint_limits, k):
        self.limits = joint_limits
        self.n = len(joint_limits)  # Number of DOFs (e.g., 5)
        self.k = k                  # Grid points per dimension
        
        # Pre-compute the 1D grids for each joint for instant lookup
        self.joint_grids = []
        for limits in self.limits:
            self.joint_grids.append(np.linspace(limits[0], limits[1], self.k))
            
    def index_to_angles(self, index_tuple):
        """
        Converts a grid index like (0, 5, 2, 1, 9) into actual physical joint angles (radians).
        """
        angles = []
        for dim in range(self.n):
            idx = index_tuple[dim]
            angles.append(self.joint_grids[dim][idx])
        return np.array(angles)

    def angles_to_index(self, angles):
        """
        Snaps physical target angles [q1, q2, q3, q4, q5] to the nearest grid index.
        """
        index_tuple = []
        for dim in range(self.n):
            # Find the closest discrete index in this joint's 1D grid
            idx = (np.abs(self.joint_grids[dim] - angles[dim])).argmin()
            index_tuple.append(idx)
        return tuple(index_tuple)

    def get_neighbors(self, index_tuple):
        """
        Returns a list of all valid, 1-connected neighboring node indices.
        """
        neighbors = []
        for dim in range(self.n):
            for step in [-1, 1]:  # Look left and right along this specific joint
                new_idx = index_tuple[dim] + step
                
                # Make sure the neighbor doesn't violate the joint limits
                if 0 <= new_idx < self.k:
                    neighbor_list = list(index_tuple)
                    neighbor_list[dim] = new_idx
                    neighbors.append(tuple(neighbor_list))
                    
        return neighbors