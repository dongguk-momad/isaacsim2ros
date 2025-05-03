from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim 관련 모듈
import omni
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api import PhysicsContext

# ROS 2 관련
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from momad_msgs.msg import ControlValue, GripperValue

import time
# Isaac Sim ROS 2 Bridge 활성화
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# 조인트 관련 설정
MAX_GRIPPER_POS = 0.025  # meter
kp = 7
ki = 0.05
kd = 0.75
kf = 0.008
max_force = 130
integral_boundary = 0.1



# kp = 5
# kd = 0.6
# kf = 0.01

class GripperController(Node):
    def __init__(self):
        super().__init__("gripper_controller")
        self.physicscontext = PhysicsContext(physics_dt=1.0 / 60.0)
        self.physicscontext.set_physics_dt(dt = 1.0 / 120, substeps=4)
        self.target_ratio = 0.0  # default target
        self.subscription = self.create_subscription(ControlValue, "/master_info", self.command_callback, 10)
        self.publisher = self.create_publisher(ControlValue, "/slave_info", 10)
        # self.pub_error = self.create_publisher(Float32, "/isaacsim_pos_error", 10)
        # self.pub_master_pos = self.create_publisher(Float32, "/master_pos", 10)

        # self.pub_test_force_p = self.create_publisher(Float32, "/force_p", 10)
        # self.pub_test_force_i = self.create_publisher(Float32, "/force_i", 10)        
        # self.pub_test_force_d = self.create_publisher(Float32, "/force_d", 10)
        # self.pub_test_force_ff = self.create_publisher(Float32, "/force_ff", 10)
        # self.pub_test_force = self.create_publisher(Float32, "/force", 10)       

        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 그리퍼 로딩
        # gripper_usd_path = "/home/user/Documents/hande.usd"
        gripper_usd_path = "/home/choiyj/Documents/hande_edited.usd"
        add_reference_to_stage(gripper_usd_path, "/World/Gripper")
        self.world.reset()
        self.gripper = Articulation("/World/Gripper")
        self.gripper.initialize()

        cube = self.world.scene.add(
                DynamicCuboid(
                    name="cube",
                    position=np.array([0.0, 0.0, 0.3]),
                    prim_path="/World/Cube",
                    scale=np.array([0.02, 0.02, 0.02]),
                    size=1.0,
                    color=np.array([0, 0, 1]),
                )
        )

        self.master_gripper_velocity = 0     
        self.gripper_target_position = 0.01

        # stabilize
        for _ in range(20):
            self.world.step(render=True)

    def command_callback(self, msg):
        self.gripper_target_position = float(np.clip(msg.gripper_state.position, 0.0, 1.0)) * MAX_GRIPPER_POS
        self.master_gripper_velocity = msg.gripper_state.velocity
        # self.master_gripper_force = msg.gripper_state.force

    def publish_gripper_state(self, position, velocity=0.0, force=0.0):
        ControllerState = ControlValue()
        GripperState = GripperValue()
        GripperState.position = float(np.clip(position / MAX_GRIPPER_POS, 0.0, 1.0))
        # GripperState.position = float(position) # 태은
        
        GripperState.velocity = float(velocity)
        GripperState.force = float(force)
        ControllerState.gripper_state = GripperState
        self.publisher.publish(ControllerState)

    def run(self):
        self.timeline.play()
        reset_needed = False
        current_error = 0.0 # 태은 추가
        integral_error = [0.0, 0.0]
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

                # 컨트롤 루프
                # force = self.gripper.get_joint_forces()[0] # N
                self.gripper.apply_action(ArticulationAction(joint_positions=[self.gripper_target_position, self.gripper_target_position], joint_indices=[0, 1]))

                pos = self.gripper.get_joint_positions()[0] # m
                vel = self.gripper.get_joint_velocities()[0] # m/s
                efforts = self.gripper.get_measured_joint_efforts()[0]
                
                self.publish_gripper_state(pos[0], vel[0], efforts[0])


                # self.pub_test_force_p.publish(Float32(data=force_p))
                # self.pub_test_force_i.publish(Float32(data=force_i))
                # self.pub_test_force_d.publish(Float32(data=force_d))            
                # self.pub_test_force_ff.publish(Float32(data=force_ff))
                # self.pub_test_force.publish(Float32(data=force))

        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()

# 실행
if __name__ == "__main__":
    rclpy.init()
    node = GripperController()
    node.run()


# slave_force : opne -> +
# master_force : open -> +