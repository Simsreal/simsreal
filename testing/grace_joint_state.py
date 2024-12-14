import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState


class TestROS2Bridge(Node):
    def __init__(self):
        super().__init__("test_ros2bridge")

        # Create the publisher. This publisher will publish a JointState message to the /joint_command topic.
        self.publisher_ = self.create_publisher(JointState, "joint_command", 10)

        # Create a JointState message
        self.joint_state = JointState()

        self.joint_state.name = [
            "pelvis",
            "right_lower_arm",
            "left_lower_arm",
            "right_shin",
            "left_shin",
        ]

        num_joints = len(self.joint_state.name)

        # make sure kit's editor is playing for receiving messages
        self.joint_state.position = np.array(
            [0.0] * num_joints, dtype=np.float64
        ).tolist()
        self.default_joints = [0.0, -1.16, -0.0, -2.3, -0.0]

        # limiting the movements to a smaller range (this is not the range of the robot, just the range of the movement
        self.max_joints = np.array(self.default_joints) + 5.0
        self.min_joints = np.array(self.default_joints) - 5.0

        # position control the robot to wiggle around each joint
        self.time_start = time.time()

        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.joint_state.header.stamp = self.get_clock().now().to_msg()

        joint_position = (
            np.sin(time.time() - self.time_start)
            * (self.max_joints - self.min_joints)
            * 0.5
            + self.default_joints
        )
        self.joint_state.position = joint_position.tolist()

        # Publish the message to the topic
        self.publisher_.publish(self.joint_state)


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    ros2_publisher = TestROS2Bridge()

    # Create and configure the executor
    executor = MultiThreadedExecutor(num_threads=2)  # You can adjust number of threads
    executor.add_node(ros2_publisher)

    try:
        # Spin the executor instead of the node directly
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        executor.shutdown()
        ros2_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
