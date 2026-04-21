import numpy as np

class LinearTrajectoryExecutor:
    def __init__(self, publish_rate_hz=50.0, time_per_segment=2.0):
        self.publish_rate_hz = publish_rate_hz
        self.time_per_segment = time_per_segment
        
        self.waypoints = []
        self.current_wp_idx = 0
        self.is_moving = False
        
        self.steps_total = 0
        self.step_current = 0
        self.current_pose = np.zeros(5)
        self.step_increment = np.zeros(5)

    def load_path(self, waypoints):
        self.waypoints = np.array(waypoints)
        if len(self.waypoints) < 2:
            return False

        self.current_wp_idx = 0
        self.current_pose = np.copy(self.waypoints[0])
        self.setup_next_segment()
        self.is_moving = True
        return True

    def setup_next_segment(self):
        start_wp = self.waypoints[self.current_wp_idx]
        end_wp = self.waypoints[self.current_wp_idx + 1]
        
        self.steps_total = int(self.time_per_segment * self.publish_rate_hz)
        self.step_current = 0
        self.step_increment = (end_wp - start_wp) / self.steps_total

    def step(self):
        """
        Calculates the next mathematical frame. 
        Returns the new joint array if moving, or None if finished/idle.
        """
        if not self.is_moving:
            return None
            
        self.current_pose += self.step_increment
        self.step_current += 1
        
        if self.step_current >= self.steps_total:
            self.current_pose = np.copy(self.waypoints[self.current_wp_idx + 1]) 
            self.current_wp_idx += 1
            
            if self.current_wp_idx >= len(self.waypoints) - 1:
                self.is_moving = False
            else:
                self.setup_next_segment()
                
        return self.current_pose.tolist()