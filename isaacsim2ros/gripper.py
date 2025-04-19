from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim 관련 모듈
import omni
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.api.objects import DynamicCuboid

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
kp = 6
kd = 0.75
kf = 0.012
max_force = 130

# kp = 5
# kd = 0.6
# kf = 0.01

class GripperController(Node):
    def __init__(self):
        super().__init__("gripper_controller")

        self.target_ratio = 0.0  # default target
        self.subscription = self.create_subscription(ControlValue, "/master_info", self.command_callback, 10)
        self.publisher = self.create_publisher(ControlValue, "/gripper_state", 10)
        self.pub_error = self.create_publisher(Float32, "/isaacsim_pos_error", 10)
        self.pub_master_pos = self.create_publisher(Float32, "/master_pos", 10)

        self.pub_test_force_p = self.create_publisher(Float32, "/force_p", 10)
        self.pub_test_force_d = self.create_publisher(Float32, "/force_d", 10)
        
        self.pub_test_force_ff = self.create_publisher(Float32, "/force_ff", 10)
        self.pub_test_force = self.create_publisher(Float32, "/force", 10)       
       
        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 그리퍼 로딩
        gripper_usd_path = "/home/user/Documents/hande.usd"
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

        # stabilize
        for _ in range(20):
            self.world.step(render=True)

    def command_callback(self, msg):
        self.target_ratio = float(np.clip(msg.gripper_state.position, 0.0, 1.0))
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
        last_error = 0.0 # 태은 추가
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
                pos = self.gripper.get_joint_positions()[0] # m
                vel = self.gripper.get_joint_velocities()[0] # m/s
                target_pos = self.target_ratio * MAX_GRIPPER_POS # m

                efforts = []
                for i in range(2):
                    error = target_pos - pos[i]
                    derror = -vel[i]
                    force_p = kp * error # p controller 
                    force_d = kd * derror # d controller
                    force_ff = -self.master_gripper_velocity*kf # ff controller
                    force = force_p + force_d + force_ff
                    # force = np.clip(force, -max_force, max_force)
                    efforts.append(force)

                
                # for i in range(2):                    
                #     current_error = target_pos - pos[i]
                #     derror = current_error - last_error
                    
                #     force = kp * current_error + kd * derror
                #     force = np.clip(force, -max_force, max_force)
                #     efforts.append(force)
                #     last_error = current_error

                # for rqt
                self.pub_error.publish(Float32(data=error)) 
                self.pub_master_pos.publish(Float32(data=target_pos)) 

                if time.time() - start_time > 0.1:
                    print(f"pos: {pos}, vel: {vel}, target: {target_pos}, efforts: {efforts}, error: {current_error}")
                    print(f"force_p: {force_p}, force_d: {force_d}. force_ff: {force_ff}, force: {force}")
                    print("#########################")
                    start_time = time.time()

                self.gripper.set_joint_efforts(efforts)
                self.publish_gripper_state(pos[0], vel[0], efforts[0])


                self.pub_test_force_p.publish(Float32(data=force_p))
                self.pub_test_force_d.publish(Float32(data=force_d))

                self.pub_test_force_ff.publish(Float32(data=force_ff))
                self.pub_test_force.publish(Float32(data=force))

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