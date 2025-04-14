from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim 관련 모듈
import omni
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.extensions import enable_extension

# ROS 2 관련
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# Isaac Sim ROS 2 Bridge 활성화
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# 조인트 관련 설정
MAX_GRIPPER_POS = 0.025  # meter
kp = 3.5
kd = 0.5
max_force = 0.03

class GripperController(Node):
    def __init__(self):
        super().__init__("gripper_controller")

        self.target_ratio = 0.0  # default target
        self.subscription = self.create_subscription(Float32, "/gripper_command", self.command_callback, 10)
        self.publisher = self.create_publisher(Float32, "/gripper_state", 10)

        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 그리퍼 로딩
        gripper_usd_path = "/home/choiyj/Documents/hande.usd"
        add_reference_to_stage(gripper_usd_path, "/World/Gripper")
        self.world.reset()
        self.gripper = Articulation("/World/Gripper")
        self.gripper.initialize()

        # stabilize
        for _ in range(20):
            self.world.step(render=True)

    def command_callback(self, msg):
        self.target_ratio = float(np.clip(msg.data, 0.0, 1.0))

    def publish_gripper_state(self, position):
        ratio = float(np.clip(position / MAX_GRIPPER_POS, 0.0, 1.0))
        self.publisher.publish(Float32(data=ratio))

    def run(self):
        self.timeline.play()
        reset_needed = False

        while simulation_app.is_running():
            self.world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.world.reset()
                    reset_needed = False

                # 컨트롤 루프
                pos = self.gripper.get_joint_positions()[0]
                vel = self.gripper.get_joint_velocities()[0]
                target_pos = self.target_ratio * MAX_GRIPPER_POS

                efforts = []
                for i in range(2):
                    error = target_pos - pos[i]
                    derror = -vel[i]
                    force = kp * error + kd * derror
                    force = np.clip(force, -max_force, max_force)
                    efforts.append(force)

                self.gripper.set_joint_efforts(efforts)
                self.publish_gripper_state(pos[0])

        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()

# 실행
if __name__ == "__main__":
    rclpy.init()
    node = GripperController()
    node.run()
