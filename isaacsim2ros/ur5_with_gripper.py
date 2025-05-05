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
from isaacsim.core.api import PhysicsContext
from isaacsim.core.api.objects import DynamicCuboid

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

class RobotarmController(Node):
    def __init__(self):
        super().__init__("robotarm_controller")
        self.physicscontext = PhysicsContext(physics_dt=1.0 / 60.0)
        self.physicscontext.set_physics_dt(dt = 1.0 / 240, substeps=8)

        self.target_position = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # default target
        self.subscription = self.create_subscription(ControlValue, "/master_info", self.command_callback, 10)
        self.publisher = self.create_publisher(ControlValue, "/slave_info", 10)

        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        ur5_usd_path = "/home/choiyj/Desktop/moma/urhand5_flatten_test.usd"
        add_reference_to_stage(ur5_usd_path, "/World")

        # SingleManipulator 생성
        self.robotarm = SingleManipulator(
            prim_path="/World/ur5",
            end_effector_prim_path="/World/ur5/wrist_3_link/flange",  # UR5의 엔드이펙터 링크 이름
            name="ur5",
        )
        self.gripper = Articulation("/World/hande")

        # 월드 초기화
        # self.world.scene.add_default_ground_plane()
        self.world.reset()

        # UR5 초기화 (필수!)
        self.robotarm.initialize()
        self.gripper.initialize()

        self.target_position = [0.0] * 6
        self.gripper_target_position = 0.0
        self.gripper_target_velocity = 0   

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
        self.gripper_target_velocity = msg.gripper_state.velocity *0.1

        # self.target_position.extend([self.target_ratio * MAX_GRIPPER_POS]*2)
        
    def publish_slave_info(self):
        msg = ControlValue()
        pos, vel, force = self.get_robot_state()
        msg.robotarm_state.position = pos[0:6]
        msg.robotarm_state.velocity = vel[0:6]
        msg.robotarm_state.force = force[0:6]
        msg.gripper_state.position = float(np.clip(pos[6] / MAX_GRIPPER_POS, 0.0, 1.0))
        msg.gripper_state.velocity = float(vel[6])
        msg.gripper_state.force = float(force[6])
        self.publisher.publish(msg)

    def get_robot_state(self):
        # Get the robot state (position, velocity, force)
        position = self.robotarm.get_joint_positions().tolist()
        velocity = self.robotarm.get_joint_velocities().tolist()
        force = self.robotarm.get_measured_joint_efforts().tolist()
        gravity = self.gripper.get_generalized_gravity_forces().tolist()[0]
        cor = self.gripper.get_coriolis_and_centrifugal_forces().tolist()[0]
        real_force = [f - g - c for f, g, c in zip(force, gravity, cor)]
        # print("gravity: ", gravity)
        # print("cor: ", cor)
        # print("force - gravity - cor: ", np.array(force) - np.array(gravity) - np.array(cor))
        return position, velocity, real_force

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

                self.robotarm.apply_action(ArticulationAction(joint_positions=[self.gripper_target_position, self.gripper_target_position], joint_velocities=[self.gripper_target_velocity*0.1, self.gripper_target_velocity*0.1], joint_indices=[6, 7]))

                self.robotarm.apply_action(ArticulationAction(joint_positions=self.target_position, 
                                           joint_indices=[0, 1, 2, 3, 4, 5]))
                self.publish_slave_info()

        # 시뮬레이션 종료
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()



if __name__ == "__main__":
    rclpy.init()
    robotarm_controller = RobotarmController()
    robotarm_controller.run()