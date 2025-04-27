from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})  # GUI를 띄우고 실행하려면 False

import omni
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import Articulation

# ROS 2 관련
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from momad_msgs.msg import ControlValue

# Isaac Sim ROS 2 Bridge 활성화
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


class RobotarmController(Node):
    def __init__(self):
        super().__init__("robotarm_controller")

        self.target_ratio = 0.0  # default target
        self.subscription = self.create_subscription(ControlValue, "/master_info", self.command_callback, 10)
        self.publisher = self.create_publisher(ControlValue, "/slave_info", 10)

        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # UR5 로딩
        asset_root = get_assets_root_path()
        ur5_usd_path = asset_root + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
        add_reference_to_stage(ur5_usd_path, "/UR5")

        self.world.reset()

        self.robotarm = SingleManipulator("/UR5")
        self.robotarm.initialize()

        # stabilize
        for _ in range(20):
            self.world.step(render=True)

    def command_callback(self, msg):
        self.target_position = msg.robotarm_state.position
        self.master_velocity = msg.robotarm_state.velocity
        self.master_force = msg.robotarm_state.force
        
    def publish_slave_info(self):
        msg = ControlValue()
        msg.robotarm_state.position = self.robotarm.get_joint_positions()
        msg.robotarm_state.velocity = self.robotarm.get_joint_velocities()
        msg.robotarm_state.force = self.robotarm.get_joint_torques()
        self.publisher.publish(msg)

    def run(self):
        self.timeline.start()
        reset_needed = False
        start_time = time.time()
        while simulation_app.is_running():
            self.world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.world.reset()
                    reset_needed = False

                self.robotarm.apply_action(ArticulationAction(joint_positions=self.target_position))
                self.publish_slave_info()

        # 시뮬레이션 종료
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()



if __name__ == "__main__":
    rclpy.init()
    robotarm_controller = RobotarmController()
    robotarm_controller.run()