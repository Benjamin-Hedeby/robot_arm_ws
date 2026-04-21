import numpy as np
import heapq
import time

class AStarPlanner:
    def __init__(self, grid, collision_func):
        self.grid = grid
        self.is_state_valid = collision_func  # Inject the collision math here

    def _calculate_distance(self, index_a, index_b):
        return np.linalg.norm(self.grid.index_to_angles(index_a) - self.grid.index_to_angles(index_b))

    def plan(self, start_angles, goal_angles, timeout=50.0):
        start_node = self.grid.angles_to_index(start_angles)
        goal_node = self.grid.angles_to_index(goal_angles)

        timer = time.time()
        
        open_set = []
        heapq.heappush(open_set, (0.0, start_node))
        
        came_from = {}
        g_score = {start_node: 0.0}
        f_score = {start_node: self._calculate_distance(start_node, goal_node)}
        visited = set()

        while open_set:
            _, current_node = heapq.heappop(open_set)
            
            if time.time() - timer > timeout:
                print(f"[A* Planner] TIMEOUT: Could not find a path within {timeout} seconds.")
                return None # Abort the search!

            if current_node == goal_node:
                raw_path = self._reconstruct_path(came_from, current_node)
                raw_path[0] = start_angles
                raw_path[-1] = goal_angles
                return raw_path

            if current_node in visited: continue
            visited.add(current_node)

            for neighbor in self.grid.get_neighbors(current_node):
                if neighbor in visited: continue
                
                neighbor_angles = self.grid.index_to_angles(neighbor)
                
                # Use the injected collision function!
                if not self.is_state_valid(neighbor_angles):
                    continue

                step_cost = self._calculate_distance(current_node, neighbor)
                tentative_g_score = g_score[current_node] + step_cost
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._calculate_distance(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _reconstruct_path(self, came_from, current_node):
        path_indices = [current_node]
        while current_node in came_from:
            current_node = came_from[current_node]
            path_indices.append(current_node)
        path_indices.reverse()
        return [self.grid.index_to_angles(idx) for idx in path_indices]