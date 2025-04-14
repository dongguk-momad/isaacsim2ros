import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import numpy as np

class GripperCommandTestNode(Node):
    def __init__(self):
        super().__init__("gripper_command_test_node")

        self.publisher = self.create_publisher(Float32, "/gripper_command", 10)

        timer_period = 0.05  # 50ms 간격 (20Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.steps = np.concatenate([
            np.linspace(0.0, 1.0, 100),
            np.linspace(1.0, 0.0, 100),
            np.linspace(0.0, 1.0, 50),
            np.linspace(1.0, 0.0, 50),
            np.linspace(0.0, 1.0, 25),
            np.linspace(1.0, 0.0, 25),
        ])
        self.index = 0
        self.get_logger().info("Gripper command test node started.")

    def timer_callback(self):
        if self.index >= len(self.steps):
            self.index = 0  # loop back

        value = float(self.steps[self.index])
        msg = Float32()
        msg.data = value
        self.publisher.publish(msg)
        self.get_logger().info(f"Publishing gripper command: {value:.3f}")
        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    node = GripperCommandTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
