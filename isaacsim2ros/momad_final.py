from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})  # GUI를 띄우고 실행하려면 False

import omni
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import Articulation
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import PhysicsContext

# ROS 2 관련
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CompressedImage
import cv2
from momad_msgs.msg import ControlValue, GripperValue

# Isaac Sim ROS 2 Bridge 활성화
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
simulation_app.set_setting("/rtx/pathtracing/enable", False)
simulation_app.set_setting("/rtx/post/dlss/enable",   True)


MAX_GRIPPER_POS = 0.025

WHEEL_RADIUS = 0.147
WHEEL_BASE = 0.645

# UR5 (shoulder → wrist)
UR5_INDICES = [
    4,  # shoulder_pan_joint
    2,  # shoulder_lift_joint
    0,  # elbow_joint
    1,  # wrist_1_joint
    3,  # wrist_2_joint
    5,  # wrist_3_joint
]

# Robotiq Hand‑E (finger L → finger R)
HANDE_INDICES = [
    6,  # robotiq_hande_left_finger_joint
    7,  # robotiq_hande_right_finger_joint
]

# Clearpath Jackal 4WD (FL → FR → RL → RR)
JACKAL_INDICES = [
    8,   # front_left_wheel
    9,   # front_right_wheel
    10,  # rear_left_wheel
    11,  # rear_right_wheel
]

WHOLE_INDICES = UR5_INDICES+HANDE_INDICES+JACKAL_INDICES
      

class RobotarmController(Node):
    def __init__(self):
        super().__init__("robotarm_controller")
        
        self.target_position = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # default target
        self.subscription = self.create_subscription(ControlValue, "/master_info", self.command_callback, 1)
        self.publisher = self.create_publisher(ControlValue, "/slave_info", 1)
        self.publisher2 = self.create_publisher(Float32MultiArray, "/force", 1)
        self.publisher_img_1 = self.create_publisher(
            CompressedImage, "/mobile_cam/rgb/compressed",  QoSProfile(depth=1)
        )
        self.publisher_img_2 = self.create_publisher(
            CompressedImage, "/hand_cam/rgb/compressed",  QoSProfile(depth=1)
        )
        self.publisher_depth_1 = self.create_publisher(
            CompressedImage, "/mobile_cam/depth/compressed",  QoSProfile(depth=1)
        )
        self.publisher_depth_2 = self.create_publisher(
            CompressedImage, "/hand_cam/depth/compressed",  QoSProfile(depth=1)
        )

        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0, physics_dt=1/240, rendering_dt=1/25)
        self.world.scene.add_default_ground_plane()

        # ur5_usd_path = "/home/choiyj/Desktop/moma/urhand5_flatten.usd"
        momad_usd_path = "/home/choiyj/Desktop/momad_test3_cam.usd"
        add_reference_to_stage(momad_usd_path, "/World")

        # SingleManipulator 생성
        self.robotarm = SingleManipulator(
            prim_path="/World/ur5",
            end_effector_prim_path="/World/ur5/wrist_3_link/flange",  # UR5의 엔드이펙터 링크 이름
            name="ur5",
        )
        self.gripper = Articulation("/World/hande")

        self.cam_mobile = Camera(
            prim_path="/World/jackal_basic/base_link/RSD455/Camera_Pseudo_Depth",
            resolution=(800, 600), 
        )

        self.cam_hand = Camera(
            prim_path="/World/hande/tool0/RSD455/Camera_Pseudo_Depth",
            resolution=(800, 600), 
        )

        # 월드 초기화
        # self.world.scene.add_default_ground_plane()
        self.world.reset()

        # UR5 초기화 (필수!)
        self.robotarm.initialize()
        self.gripper.initialize()
        self.cam_hand.initialize()
        self.cam_mobile.initialize()

        self.cam_hand.add_distance_to_image_plane_to_frame()
        self.cam_mobile.add_distance_to_image_plane_to_frame()

        self.robotarm_target_position = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
        self.robotarm_target_velocity = [0.0] * 6
        
        self.gripper_target_position = [0.0] * 2
        self.gripper_target_velocity = [0.0] * 2 

        self.mobile_target_angular = 0.0
        self.mobile_target_linear = 0.0

        self.robotarm.set_joint_positions(self.robotarm_target_position, UR5_INDICES)
        # stabilize
        for _ in range(40):
            self.world.step(render=True)

        cube1 = self.world.scene.add(
                DynamicCuboid(
                    name="cube1",
                    position=np.array([0.9, -0.1, 0.9]),
                    prim_path="/World/Cube1",
                    scale=np.array([0.03, 0.03, 0.03]),
                    size=1.0,
                    color=np.array([0, 1, 0]),
                )
        )

        cube2 = self.world.scene.add(
                DynamicCuboid(
                    name="cube2",
                    position=np.array([0.9, 0.0, 0.4]),
                    prim_path="/World/Cube2",
                    scale=np.array([0.5, 0.5, 0.8]),
                    size=1.0,
                    color=np.array([0.05, 0.1, 0.1]),
                )
        )

    def publish_images(self, rgb_hand, rgb_mobile, depth_hand, depth_mobile):
        """RGB/Depth × 2 프레임을 JPEG로 압축해 퍼블리시"""
        t_total_start = time.perf_counter() # 함수 전체 시작 시간

        t_get_data_start = time.perf_counter()
        # 이미지 데이터는 이미 인자로 받아오므로, 여기서는 준비 과정이 있다면 측정
        # 예: frames_raw = [rgb_mobile, rgb_hand, depth_mobile, depth_hand]
        t_get_data_end = time.perf_counter()
        get_data_time = t_get_data_end - t_get_data_start

        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60] # self.jpeg_quality 사용 (예: 80)
        pub_tasks = [
            (rgb_mobile, self.publisher_img_1, "mobile_cam", "bgr"),
            (rgb_hand, self.publisher_img_2, "hand_cam", "bgr"),
            (depth_mobile, self.publisher_depth_1, "mobile_cam", "gray"),
            (depth_hand, self.publisher_depth_2, "hand_cam", "gray"),
        ]

        now = self.get_clock().now().to_msg()

        total_preprocessing_time = 0
        total_imencode_time = 0
        total_ros_publish_time = 0

        for frame, pub, fid, mode in pub_tasks:
            t_prep_start = time.perf_counter()
            if mode == "bgr":
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:  # "gray" (Depth)
                frame8 = np.clip(frame, 0, 5.0) * 51
                frame_processed = frame8.astype(np.uint8)
            t_prep_end = time.perf_counter()
            total_preprocessing_time += (t_prep_end - t_prep_start)

            t_encode_start = time.perf_counter()
            ok, buf = cv2.imencode(".jpg", frame_processed, enc_param)
            t_encode_end = time.perf_counter()
            total_imencode_time += (t_encode_end - t_encode_start)

            if not ok:
                self.get_logger().error(f"JPEG encode failed for {fid}")
                continue

            t_ros_pub_start = time.perf_counter()
            msg = CompressedImage()
            msg.header.stamp = now
            msg.header.frame_id = fid
            msg.format = "jpeg"
            msg.data = memoryview(buf).tobytes()
            pub.publish(msg)
            t_ros_pub_end = time.perf_counter()
            total_ros_publish_time += (t_ros_pub_end - t_ros_pub_start)
        
        t_total_end = time.perf_counter() # 함수 전체 종료 시간
        
        # 상세 시간 로깅 (매번 로깅하면 너무 많으니, 가끔 확인하거나 평균값을 내는 용도로 사용)
        # if self.loop_counter % 30 == 0: # 예: 30프레임마다 한 번씩 로깅
        self.get_logger().info(
            f"[PubProfile] Total: {t_total_end - t_total_start:.5f}s | "
            f"GetData: {get_data_time:.5f}s | Preproc: {total_preprocessing_time:.5f}s | "
            f"Encode: {total_imencode_time:.5f}s | ROSPub: {total_ros_publish_time:.5f}s"
        )

        
    def command_callback(self, msg):
        latency = time.time() - msg.stamp
        print("ws latency: ", latency)
        # pass
        self.robotarm_target_position = msg.robotarm_state.position
        self.robotarm_target_velocity = msg.robotarm_state.velocity
        
        for i in range(len(self.robotarm_target_position)):
            self.robotarm_target_position[i] = np.clip(self.robotarm_target_position[i]*np.pi/180, -3.14, 3.14)

        self.gripper_target_position = msg.gripper_state.position
        self.gripper_target_velocity = msg.gripper_state.velocity

        for i in range(len(self.gripper_target_position)):
            self.gripper_target_position[i] = np.clip(self.gripper_target_position[i], 0, 1) * MAX_GRIPPER_POS

        self.mobile_target_linear = msg.mobile_state.linear_velocity
        self.mobile_target_angular = msg.mobile_state.angular_velocity
        
    def publish_slave_info(self):
        msg = ControlValue()
        pos, vel, force, linear_vel, angular_vel, force_sensor = self.get_robot_state()
        msg.robotarm_state.position = pos[0:6]
        msg.robotarm_state.velocity = vel[0:6]
        msg.robotarm_state.force = force[0:6]
        msg.gripper_state.position = [np.clip(p / MAX_GRIPPER_POS, 0.0, 1.0) for p in pos[6:8]]
        msg.gripper_state.velocity = vel[6:8]
        msg.gripper_state.force = force[6:8]

        msg.mobile_state.linear_velocity = float(linear_vel)
        msg.mobile_state.angular_velocity = float(angular_vel)

        # force3(x, y, z), torque3(x, y, z)
        msg.force_torque = force_sensor[6]

        msg.stamp = time.time()
        self.publisher.publish(msg)

    def get_robot_state(self):
        # Get the gripper state (position, velocity, force)
        self.position = self.robotarm.get_joint_positions(joint_indices=WHOLE_INDICES).tolist()
        self.velocity = self.robotarm.get_joint_velocities(joint_indices=WHOLE_INDICES).tolist()
        force = self.robotarm.get_measured_joint_efforts(joint_indices=WHOLE_INDICES).tolist()
        gravity = self.gripper.get_generalized_gravity_forces(joint_indices=WHOLE_INDICES).tolist()[0]
        cor = self.gripper.get_coriolis_and_centrifugal_forces(joint_indices=WHOLE_INDICES).tolist()[0]
        self.real_force = [f - g - c for f, g, c in zip(force, gravity, cor)]

        self.force_sensor = self.robotarm.get_measured_joint_forces(joint_indices=WHOLE_INDICES).tolist()

        omega_L = np.mean([self.velocity[JACKAL_INDICES[0]], self.velocity[JACKAL_INDICES[2]]])
        omega_R = np.mean([self.velocity[JACKAL_INDICES[1]], self.velocity[JACKAL_INDICES[3]]])

        self.linear_vel = WHEEL_RADIUS * 0.5 * (omega_L + omega_R)
        self.angular_vel = WHEEL_RADIUS / WHEEL_BASE * (omega_R - omega_L)

        return self.position, self.velocity, self.real_force, self.linear_vel, self.angular_vel, self.force_sensor
    
    def differential_controller_skid_steer(self, target_linear_vel, target_angular_vel):
        left_correctionF = 0.08
        right_correctionF = 0.0
        left_correctionR = 0.03
        right_correctionR = 0.06

        # base speed
        chi = 1.4 # experimentally tuned
        
        effective_wheelbase = WHEEL_BASE * chi
        left_base_speed = ((2 * target_linear_vel) - (target_angular_vel * effective_wheelbase)) / (2 * WHEEL_RADIUS)
        right_base_speed = ((2 * target_linear_vel) + (target_angular_vel * effective_wheelbase)) / (2 * WHEEL_RADIUS)


        # final wheel speeds
        front_left_speed = left_base_speed - left_correctionF * target_angular_vel
        rear_left_speed = front_left_speed - left_correctionR * target_angular_vel  
        front_right_speed = right_base_speed + right_correctionF * target_angular_vel
        rear_right_speed = front_right_speed + right_correctionR * target_angular_vel

        # 속도 제한
        max_speed = 1000.0  # 최대 허용 속도
        front_left_speed = np.clip(front_left_speed, -max_speed, max_speed)
        front_right_speed = np.clip(front_right_speed, -max_speed, max_speed)
        rear_left_speed = np.clip(rear_left_speed, -max_speed, max_speed)
        rear_right_speed = np.clip(rear_right_speed, -max_speed, max_speed)

        return ArticulationAction(
            joint_velocities=[front_left_speed, front_right_speed, rear_left_speed, rear_right_speed],
            joint_indices=JACKAL_INDICES
        )
        
    def run(self):
        self.timeline.play()
        reset_needed = False

        while simulation_app.is_running():
            t0 = time.perf_counter()
            self.world.step(render=True)
            t_step = time.perf_counter()
            rclpy.spin_once(self, timeout_sec=0.0)
            t_ros = time.perf_counter()

            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.world.reset()
                    reset_needed = False
                self.get_robot_state()

                self.robotarm.apply_action(ArticulationAction(joint_positions=self.gripper_target_position, joint_velocities=self.gripper_target_velocity, joint_indices=HANDE_INDICES))

                # self.robotarm.apply_action(ArticulationAction(joint_positions=self.robotarm_target_position, joint_velocities=self.robotarm_target_velocity, joint_indices=UR5_INDICES))
                self.robotarm.apply_action(ArticulationAction(joint_positions=self.robotarm_target_position, joint_indices=UR5_INDICES))

                self.robotarm.apply_action(self.differential_controller_skid_steer(self.mobile_target_linear, self.mobile_target_angular))
                # self.robotarm.apply_action(self.differential_controller_skid_steer(0.0, 0.5))

                rgb_hand = self.cam_hand.get_rgb()
                rgb_mobile = self.cam_mobile.get_rgb()
                depth_hand = self.cam_hand.get_depth()
                depth_mobile = self.cam_mobile.get_depth()
                self.publish_images(rgb_hand, rgb_mobile, depth_hand, depth_mobile) 
                # print("depth_hand: ", depth_mobile.shape)
                self.publish_slave_info()
                t_pub = time.perf_counter()

            print(f"dt step={t_step-t0:.4f}s  ros={t_ros-t_step:.4f}s  "
                  f"pub={t_pub-t_ros:.4f}s  total={t_pub-t0:.4f}s")

        # 시뮬레이션 종료
        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()



if __name__ == "__main__":
    rclpy.init()
    robotarm_controller = RobotarmController()
    robotarm_controller.run()