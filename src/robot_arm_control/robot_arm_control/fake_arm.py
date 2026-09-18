"""Stand-in for the arm hardware and camera, for testing the task controller
without the Raspberry Pi, ODrives or OAK-D.

It plays two roles:
  * the Pi's ros2_control stack: takes joint commands, publishes joint states
    (moving each joint at a limited speed, like the real motors), and
  * the camera + detections_republish: when triggered, reports a weed at a
    configured base-frame position once the camera is over it.

Topic names are relative, so run it in the 'arm' namespace to match the real
system (arm_bringup.launch.py fake_hardware:=true does this).

Failure modes are read live, so they can be switched between cycles, e.g.
  ros2 param set /arm/fake_arm hard_floor_z -0.49   # stone under the weed
  ros2 param set /arm/fake_arm max_detections 0     # no weed at all
  ros2 param set /arm/fake_arm max_detections 2     # weed lost before FINAL_SCAN
max_detections counts per search, i.e. from each sweep trigger.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32, String
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState

from .ForwardKinematics import forward_kinematics
from .VisionTransformV2 import transform_camera_to_base
from .task_controllerV2 import PBVSTaskController

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
GRIPPER_OPEN = 0.52
GRIPPER_CLOSED = -0.65
RATE_HZ = 100.0


def with_joint6(joints):
    # The kinematics code expects 6 joints; the physical arm has 5 (joint 6 is always 0).
    return list(joints) + [0.0]


class FakeArm(Node):
    def __init__(self):
        super().__init__('fake_arm')

        # Default weed position is taken from a real logged success in the arm's CSVs.
        self.declare_parameter('weed_x', 0.50)
        self.declare_parameter('weed_y', -0.03)
        self.declare_parameter('weed_z', -0.48)
        self.declare_parameter('view_radius', 0.15)     # m, camera sees the weed within this horizontal distance
        self.declare_parameter('camera_fps', 20.0)      # a trigger of N samples takes N / fps seconds
        # Per search (reset at each sweep trigger): -1 = unlimited, 0 = no weed, 2 = lost before FINAL_SCAN
        self.declare_parameter('max_detections', -1)
        self.declare_parameter('hard_floor_z', -10.0)   # m, the TCP can't go below this (a stone); -10 = no floor
        self.declare_parameter('max_joint_speed', 1.5)  # rad/s

        self.joints = [0.0] * 5
        self.target = [0.0] * 5
        self.gripper = GRIPPER_OPEN

        self.trigger_samples = 0      # > 0 while a measurement is requested
        self.seen_since = None        # time the weed came into view during the current measurement
        self.detections_sent = 0

        self.create_subscription(Float64MultiArray, 'arm_controller/commands', self.on_command, 10)
        self.create_subscription(String, 'gripper_open_close_cmd', self.on_gripper, 10)
        self.create_subscription(Int32, 'trigger_measurement', self.on_trigger, 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.weed_pub = self.create_publisher(Point, 'weed_location_cam_frame', 10)

        self.create_timer(1.0 / RATE_HZ, self.step)
        self.get_logger().info('Fake arm + camera running.')

    def p(self, name):
        return self.get_parameter(name).value

    def tcp(self, joints):
        return forward_kinematics(with_joint6(joints))[:3, 3]

    # ---------------- arm ----------------
    def on_command(self, msg):
        if len(msg.data) != 5:
            self.get_logger().warn(f'Expected 5 joint values, got {len(msg.data)}')
            return
        command = list(msg.data)
        if self.tcp(command)[2] < self.p('hard_floor_z'):
            self.get_logger().warn('Blocked: command would push the tool below the hard floor.',
                                   throttle_duration_sec=1.0)
            return
        self.target = command

    def on_gripper(self, msg):
        command = msg.data.strip().lower()
        if command in ('open', 'close'):
            self.gripper = GRIPPER_OPEN if command == 'open' else GRIPPER_CLOSED
            self.get_logger().info(f'Gripper: {command}')

    def move_joints(self):
        max_step = self.p('max_joint_speed') / RATE_HZ
        self.joints = [j + float(np.clip(t - j, -max_step, max_step))
                       for j, t in zip(self.joints, self.target)]

    # ---------------- camera ----------------
    def on_trigger(self, msg):
        if msg.data == PBVSTaskController.TRIGGER_SWEEP:
            self.detections_sent = 0  # a sweep starts a new search
        self.trigger_samples = max(0, msg.data)
        self.seen_since = None

    def weed(self):
        return np.array([self.p('weed_x'), self.p('weed_y'), self.p('weed_z')])

    def weed_in_view(self):
        limit = self.p('max_detections')
        if 0 <= limit <= self.detections_sent:
            return False
        camera = np.array(transform_camera_to_base(0.0, 0.0, 0.0, with_joint6(self.joints)))
        return np.linalg.norm(camera[:2] - self.weed()[:2]) <= self.p('view_radius')

    def weed_in_camera_frame(self):
        # transform_camera_to_base is affine for a fixed arm pose, so measure it
        # and invert it: the task controller then transforms this point straight
        # back to the configured weed position. No IMU is published, so both
        # sides use the forward-kinematics camera orientation.
        joints = with_joint6(self.joints)
        origin = np.array(transform_camera_to_base(0.0, 0.0, 0.0, joints))
        axes = np.column_stack([
            np.array(transform_camera_to_base(*unit, joints)) - origin
            for unit in np.eye(3)
        ])
        return np.linalg.solve(axes, self.weed() - origin)

    def update_camera(self):
        if self.trigger_samples <= 0:
            return
        if not self.weed_in_view():
            self.seen_since = None
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self.seen_since is None:
            self.seen_since = now
        elif now - self.seen_since >= self.trigger_samples / self.p('camera_fps'):
            x, y, z = self.weed_in_camera_frame()
            self.weed_pub.publish(Point(x=float(x), y=float(y), z=float(z)))
            self.detections_sent += 1
            self.get_logger().info(f'Camera: weed reported ({self.trigger_samples}-sample measurement).')
            self.trigger_samples = 0

    # ---------------- loop ----------------
    def step(self):
        self.move_joints()
        self.update_camera()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES + ['gripper1']
        msg.position = [float(j) for j in self.joints] + [self.gripper]
        self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
