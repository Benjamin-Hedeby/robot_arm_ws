import numpy as np

class StringPuller:
    def __init__(self, collision_func):
        self.is_state_valid = collision_func

    def _is_line_collision_free(self, start_angles, end_angles):
        steps = 10
        for i in range(1, steps):
            fraction = i / float(steps)
            test_angles = start_angles + (end_angles - start_angles) * fraction
            
            if not self.is_state_valid(test_angles):
                return False 
        return True 

    def smooth(self, raw_path):
        if len(raw_path) <= 2:
            return raw_path

        smoothed_path = [raw_path[0]]
        current_idx = 0

        while current_idx < len(raw_path) - 1:
            furthest_safe_idx = current_idx + 1
            
            for check_idx in range(current_idx + 2, len(raw_path)):
                if self._is_line_collision_free(raw_path[current_idx], raw_path[check_idx]):
                    furthest_safe_idx = check_idx
                else:
                    break
                    
            smoothed_path.append(raw_path[furthest_safe_idx])
            current_idx = furthest_safe_idx

        return smoothed_path