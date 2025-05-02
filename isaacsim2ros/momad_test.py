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
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController

# ROS 2 관련
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from momad_msgs.msg import ControlValue, GripperValue

# Isaac Sim ROS 2 Bridge 활성화
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
MAX_GRIPPER_POS = 0.025
kp = 7
ki = 0.05
kd = 0.75
kf = 0.008
max_force = 130
integral_boundary = 0.1

# UR5 (shoulder → wrist)
ur5_indices = [
    4,  # shoulder_pan_joint
    2,  # shoulder_lift_joint
    0,  # elbow_joint
    1,  # wrist_1_joint
    3,  # wrist_2_joint
    5,  # wrist_3_joint
]

# Robotiq Hand‑E (finger L → finger R)
hande_indices = [
    6,  # robotiq_hande_left_finger_joint
    7,  # robotiq_hande_right_finger_joint
]

# Clearpath Jackal 4WD (FL → FR → RL → RR)
jackal_indices = [
    8,   # front_left_wheel
    9,   # front_right_wheel
    10,  # rear_left_wheel
    11,  # rear_right_wheel
]
      

class RobotarmController(Node):
    def __init__(self):
        super().__init__("robotarm_controller")

        self.target_position = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # default target
        self.subscription = self.create_subscription(ControlValue, "/master_info", self.command_callback, 10)
        self.publisher = self.create_publisher(ControlValue, "/slave_info", 10)

        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # ur5_usd_path = "/home/choiyj/Desktop/moma/urhand5_flatten.usd"
        momad_usd_path = "/home/choiyj/Desktop/momad_test.usd"
        add_reference_to_stage(momad_usd_path, "/World")

        # SingleManipulator 생성
        self.robotarm = SingleManipulator(
            prim_path="/World/ur5",
            end_effector_prim_path="/World/ur5/wrist_3_link/flange",  # UR5의 엔드이펙터 링크 이름
            name="ur5",
        )
        self.gripper = Articulation("/World/hande")

        self.diff_controller = DifferentialController(
            name="diff_controller",
            wheel_dof_indices=jackal_indices,
            wheel_radius=0.098,
            wheel_base=0.43,
            max_linear_speed = 1.0,
            max_angular_speed = 1.0,
            max_wheel_speed = 30.0,
        )
        # 월드 초기화
        # self.world.scene.add_default_ground_plane()
        self.world.reset()

        # UR5 초기화 (필수!)
        self.robotarm.initialize()
        self.gripper.initialize()

        self.target_position = [0.0] * 6
        self.gripper_target_position = 0.0
        self.master_gripper_velocity = 0   

        cube = self.world.scene.add(
                DynamicCuboid(
                    name="cube",
                    position=np.array([0.3, 0.3, 0.3]),
                    prim_path="/World/Cube",
                    scale=np.array([0.03, 0.03, 0.03]),
                    size=1.0,
                    color=np.array([0, 0, 1]),
                )
        )

        # stabilize
        for _ in range(20):
            self.world.step(render=True)

    def command_callback(self, msg):
        # pass
        self.target_position = msg.robotarm_state.position
        self.master_velocity = msg.robotarm_state.velocity
        self.master_force = msg.robotarm_state.force
        for i in range(len(self.target_position)):
            self.target_position[i] = np.clip(self.target_position[i]*np.pi/180, -3.14, 3.14)

        self.gripper_target_position = float(np.clip(msg.gripper_state.position, 0.0, 1.0)) * MAX_GRIPPER_POS
        self.master_gripper_velocity = msg.gripper_state.velocity

        # self.target_position.extend([self.target_ratio * MAX_GRIPPER_POS]*2)
        
    def publish_slave_info(self, position, velocity=0.0, force=0.0):
        msg = ControlValue()
        msg.robotarm_state.position = self.robotarm.get_joint_positions()
        msg.robotarm_state.velocity = self.robotarm.get_joint_velocities()
        msg.robotarm_state.force = self.robotarm.get_joint_torques()
        ControllerState = ControlValue()
        GripperState = GripperValue()
        GripperState.position = float(np.clip(position / MAX_GRIPPER_POS, 0.0, 1.0))
        # GripperState.position = float(position) # 태은
        
        GripperState.velocity = float(velocity)
        GripperState.force = float(force)
        ControllerState.gripper_state = GripperState
        self.publisher.publish(msg)

    def get_gripper_state(self):
        # Get the gripper state (position, velocity, force)
        position = self.robotarm.get_joint_positions()[6:8] # m
        velocity = self.robotarm.get_joint_velocities()[6:8] # m/s
        return position, velocity

    def run(self):
        self.timeline.play()
        reset_needed = False
        start_time = time.time()
        integral_error = [0.0, 0.0]
        while simulation_app.is_running():
            self.world.step(render=True)
            rclpy.spin_once(self, timeout_sec=0.0)

            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.world.reset()
                    reset_needed = False

                self.robotarm.apply_action(ArticulationAction([self.gripper_target_position], joint_indices=hande_indices))

                self.robotarm.apply_action(ArticulationAction(joint_positions=self.target_position, 
                                           joint_indices=ur5_indices))
                self.robotarm.apply_action(self.diff_controller.forward([0.1, 0.8]))
                # self.publish_slave_info()

        # 시뮬레이션 종료
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()



if __name__ == "__main__":
    rclpy.init()
    robotarm_controller = RobotarmController()
    robotarm_controller.run()