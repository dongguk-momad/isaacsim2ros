from omni.isaac.kit import SimulationApp

config = {
    "headless": False,
    "renderer": "RaytracedLighting", # "Raytracer" -> "RaytracedLighting" (표준 옵션명)
    "anti_aliasing": 0, # 나중에 코드에서 DLSS로 변경됨
}
simulation_app = SimulationApp(config)  # GUI를 띄우고 실행하려면 False

import omni
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import RigidPrim
from isaacsim.core.api import PhysicsContext

# ROS 2 관련
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # QoSProfile 등 임포트
from std_msgs.msg import Float32, Float32MultiArray, Header # 신호용으로 Header 메시지 사용
# from sensor_msgs.msg import Image # SHM 사용으로 더 이상 필요 없음
# import cv2 # Isaac Sim 측에서는 cv2 직접 사용 안 함 (NumPy 바로 사용)
# from cv_bridge import CvBridge # SHM 사용으로 더 이상 필요 없음
from momad_msgs.msg import ControlValue, GripperValue # 사용자 정의 메시지

# 공유 메모리 관련
from multiprocessing import shared_memory
import signal # SHM 정리를 위해 추가
import atexit # SHM 정리를 위해 추가

# Isaac Sim ROS 2 Bridge 활성화
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
simulation_app.set_setting("/rtx/post/aa/op", 3)
simulation_app.set_setting("/rtx-defaults/post/dlss/execMode", 0)
simulation_app.set_setting("/rtx/post/dlss/execMode", 0)
simulation_app.set_setting("/rtx-transient/dlssg/enabled", True)


MODE = "PEG_IN_HOLE"  # "PEG_IN_HOLE" or "ONLY_ROBOT"

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

# --- 공유 메모리 설정 ---
# 이미지 크기 및 데이터 타입 (FastAPI 서버와 동일하게 유지해야 함)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
RGB_CHANNELS = 3
RGB_DTYPE = np.uint8
DEPTH_CHANNELS = 1 # Depth는 단일 채널
DEPTH_DTYPE = np.float32 # Isaac Sim Camera의 get_depth()는 보통 float32 (미터 단위)

# 각 이미지에 대한 SHM 정보 (이름, 크기, 형태, 타입)
# 이 이름들은 FastAPI 서버에서도 동일하게 사용해야 합니다.
SHM_CONFIG = {
    "mobile_rgb": {"name": "shm_mobile_rgb", "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS), "dtype": RGB_DTYPE},
    "mobile_depth": {"name": "shm_mobile_depth", "shape": (IMAGE_HEIGHT, IMAGE_WIDTH), "dtype": DEPTH_DTYPE}, # Depth는 채널 축 없음
    "hand_rgb": {"name": "shm_hand_rgb", "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, RGB_CHANNELS), "dtype": RGB_DTYPE},
    "hand_depth": {"name": "shm_hand_depth", "shape": (IMAGE_HEIGHT, IMAGE_WIDTH), "dtype": DEPTH_DTYPE},
}

# 각 SHM 세그먼트의 바이트 크기 계산
for key in SHM_CONFIG:
    config_item = SHM_CONFIG[key]
    config_item["size"] = int(np.prod(config_item["shape"]) * np.dtype(config_item["dtype"]).itemsize)

class RobotarmController(Node):
    def __init__(self):
        super().__init__("robotarm_controller_shm") # 노드 이름 변경 (선택적)
        
        self.target_position = [0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.subscription = self.create_subscription(ControlValue, "/master_info", self.command_callback, 1)
        self.publisher = self.create_publisher(ControlValue, "/slave_info", 1)
        self.publisher2 = self.create_publisher(Float32MultiArray, "/force", 1)

        # --- SHM 및 신호용 Publisher 초기화 ---
        self.shm_segments = {} # 생성된 SHM 객체 저장
        self.shm_np_arrays = {} # SHM에 매핑된 NumPy 배열 저장
        self._init_shared_memory()

        # 이미지가 SHM에 준비되었음을 알리는 신호용 Publisher (std_msgs/Header 사용)
        # Header 메시지의 stamp 필드에 이미지 캡처 시간을 기록
        # frame_id 필드에 "new_images_ready"와 같은 문자열을 넣어 신호로 사용 가능
        self.image_signal_publisher = self.create_publisher(Header, "/isaac_image_signal", 10)

        # Isaac Sim World 초기화
        self.timeline = omni.timeline.get_timeline_interface()
        # physics_dt를 이전 FPS(15~18)보다 훨씬 높게 설정 (예: 60Hz, 120Hz, 또는 그 이상)
        # FastAPI 처리 루프가 약 20~25FPS (0.04~0.05초 간격)이므로, Isaac Sim은 더 빠르게 이미지를 생성할 수 있어야 함
        self.world = World(stage_units_in_meters=1.0) # 예: 60Hz

        momad_usd_path = "/home/choiyj/Desktop/momad_test3_cam_test.usd" # USD 경로 확인 필요
        add_reference_to_stage(momad_usd_path, "/World/robot")

        referenced_asset_prim = XFormPrim(
            prim_paths_expr="/World/robot",
            translations=np.array([[0.0, 0.0, 1.05],[0.0, 0.0, 1.05],[0.0, 0.0, 1.05]]),
        )

        self.robotarm = SingleManipulator(
            prim_path="/World/robot/ur5",
            end_effector_prim_path="/World/robot/ur5/wrist_3_link/flange",
            name="ur5",
        )
        self.gripper = Articulation("/World/robot/hande")

        self.cam_mobile = Camera(
            prim_path="/World/robot/jackal_basic/base_link/RSD455/Camera_Pseudo_Depth",
            resolution=(IMAGE_WIDTH, IMAGE_HEIGHT), 
        )
        self.cam_hand = Camera(
            prim_path="/World/robot/hande/tool0/RSD455/Camera_Pseudo_Depth",
            resolution=(IMAGE_WIDTH, IMAGE_HEIGHT), 
        )
        self.world.scene.add_default_ground_plane()
        self.world.reset()

        self.robotarm.initialize()
        self.gripper.initialize()
        self.cam_hand.initialize()
        self.cam_mobile.initialize()
 
        self.cam_hand.add_distance_to_image_plane_to_frame() 
        self.cam_mobile.add_distance_to_image_plane_to_frame()

        self.robotarm_target_position = [0.0, -1.0, 1.3, 0.0, 0.0, 0.0]
        self.robotarm_target_velocity = [0.0] * 6
        self.gripper_target_position = [0.0] * 2
        self.gripper_target_velocity = [0.0] * 2 
        self.mobile_target_angular = 0.0
        self.mobile_target_linear = 0.0

        self.robotarm.set_joint_positions(self.robotarm_target_position, UR5_INDICES)
        for _ in range(40): # 안정화 시간
            self.world.step(render=True)

        if MODE == "PEG_IN_HOLE":
            self.add_object_peginhole()

        # 프로그램 종료 시 SHM 정리 등록
        atexit.register(self._cleanup_shared_memory)
        # SIGINT, SIGTERM 등 주요 종료 시그널에 대한 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info("RobotarmController (SHM version) initialized.")

    def _init_shared_memory(self):
        self.get_logger().info("Initializing shared memory segments...")
        for key, config_item in SHM_CONFIG.items():
            shm_name = config_item["name"]
            shm_size = config_item["size"]
            shm_shape = config_item["shape"]
            shm_dtype = config_item["dtype"]
            try:
                # 기존에 같은 이름의 SHM이 남아있을 수 있으므로 시도 후, 없으면 생성
                shm = shared_memory.SharedMemory(name=shm_name, create=False, size=shm_size)
                self.get_logger().info(f"Attached to existing SHM: {shm_name}")
            except FileNotFoundError:
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=shm_size)
                self.get_logger().info(f"Created new SHM: {shm_name} with size {shm_size} bytes")
            
            self.shm_segments[key] = shm
            # SHM 버퍼를 NumPy 배열로 매핑
            self.shm_np_arrays[key] = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
        self.get_logger().info("Shared memory segments initialized/attached.")

    def _cleanup_shared_memory(self):
        self.get_logger().info("Cleaning up shared memory segments...")
        for key, shm in self.shm_segments.items():
            try:
                shm.close() # SHM 객체 닫기
                # 생성한 프로세스에서만 unlink 수행 (또는 마지막 detach 시)
                # 여기서는 이 클래스가 생성자라고 가정하고 unlink 시도
                shm.unlink() 
                self.get_logger().info(f"Closed and unlinked SHM: {SHM_CONFIG[key]['name']}")
            except FileNotFoundError:
                self.get_logger().warn(f"SHM segment {SHM_CONFIG[key]['name']} already unlinked or not found during cleanup.")
            except Exception as e:
                self.get_logger().error(f"Error cleaning up SHM {SHM_CONFIG[key]['name']}: {e}")
        self.shm_segments.clear()
        self.shm_np_arrays.clear()

    def _signal_handler(self, signum, frame):
        self.get_logger().warn(f"Received signal {signum}. Cleaning up and shutting down...")
        # self._cleanup_shared_memory() # atexit으로 이미 등록됨
        # rclpy.shutdown() 등은 run 메소드의 finally 블록이나 메인에서 처리
        # 여기서는 simulation_app을 안전하게 종료하는 것을 돕거나, 플래그 설정
        if simulation_app.is_running():
             simulation_app.close() # 시뮬레이션 앱 종료 요청
        # sys.exit(0) # 필요시 강제 종료 (보통은 rclpy spin 종료 후 자연스럽게)


    def add_object_peginhole(self):
        hole_usd_path = "/home/choiyj/Desktop/hole_o_30.usd"
        add_reference_to_stage(hole_usd_path, "/World/hole_o_30")
        hole = RigidPrim(
            prim_paths_expr="/World/hole_o_30",                
            name="hole_o_30",
            positions=np.array([[0.95, 0.05, 0.00]]),
            scales=[np.ones(3) * 1.0]
        )

        peg_usd_path = "/home/choiyj/Desktop/peg_o_30.usd"
        add_reference_to_stage(peg_usd_path, "/World/peg_o_30")
        peg = RigidPrim(
            prim_paths_expr="/World/peg_o_30",
            name="peg_o_30",
            positions=np.array([[0.95, 0.05, 0.8]]),
            scales=[np.ones(3) * 1.0],

        )


    def write_images_to_shm_and_signal(self, rgb_hand_np, rgb_mobile_np, depth_hand_np, depth_mobile_np):
        """RGB/Depth 이미지를 공유 메모리에 쓰고 신호를 보냅니다."""
        try:
            # 데이터 SHM에 복사
            # 형상과 타입이 일치해야 함
            self.shm_np_arrays["hand_rgb"][:] = rgb_hand_np
            self.shm_np_arrays["mobile_rgb"][:] = rgb_mobile_np
            self.shm_np_arrays["hand_depth"][:] = depth_hand_np
            self.shm_np_arrays["mobile_depth"][:] = depth_mobile_np

            # 신호용 메시지 발행
            signal_msg = Header()
            signal_msg.stamp = self.get_clock().now().to_msg()
            signal_msg.frame_id = "new_images_ready" # 어떤 신호인지 나타내는 ID
            self.image_signal_publisher.publish(signal_msg)
            # self.get_logger().info("Images written to SHM and signal published.", throttle_duration_sec=1.0)

        except Exception as e:
            self.get_logger().error(f"Error in write_images_to_shm_and_signal: {e}")
        
    def command_callback(self, msg):
        # ... (이전과 동일, 단 msg.stamp은 float이므로 ROS 시간과 직접 비교 어려움)
        # ws_latency_ms = (time.time() - msg.stamp) * 1000 # time.time()은 시스템 시간
        # self.get_logger().info(f"WebSocket command latency: {ws_latency_ms:.2f} ms", อื่นๆ)

        robotarm_target_position_deg = msg.robotarm_state.position
        self.robotarm_target_velocity = msg.robotarm_state.velocity
        robotarm_target_position_deg[5] *=-1
        # robotarm_target_position_deg[4] -= 720 # 이 로직은 각도 범위에 따라 필요성 재검토
        
        for i in range(len(robotarm_target_position_deg)):
            self.robotarm_target_position[i] = np.clip(np.deg2rad(robotarm_target_position_deg[i]), -np.pi, np.pi)
        
        self.gripper_target_position = msg.gripper_state.position
        self.gripper_target_velocity = [msg.gripper_state.velocity[0]]*2

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

        # FPS 계산용 변수 (선택적)
        # image_processed_count = 0
        # fps_log_start_time = time.perf_counter()
        # log_interval = 5.0 # 초

        while simulation_app.is_running() and rclpy.ok(): # rclpy.ok() 추가
            # t0 = time.perf_counter() # 상세 프로파일링용
            self.world.step(render=True)
            # t_step = time.perf_counter()
            rclpy.spin_once(self, timeout_sec=0.0) # ROS 이벤트 처리 (매우 짧은 타임아웃)
            # t_ros = time.perf_counter()

            if self.world.is_stopped() and not reset_needed:
                reset_needed = True
            if self.world.is_playing():
                if reset_needed:
                    self.world.reset()
                    # SHM 재초기화는 필요 없음 (프로세스 재시작 시에만)
                    self.robotarm.set_joint_positions(self.robotarm_target_position, UR5_INDICES) # 리셋 후 초기 자세
                    for _ in range(40): self.world.step(render=True) # 안정화
                    reset_needed = False
                
                self.get_robot_state() # 다른 데이터 업데이트

                # 로봇 제어 액션 적용
                self.robotarm.apply_action(ArticulationAction(joint_positions=self.gripper_target_position, joint_velocities=self.gripper_target_velocity, joint_indices=HANDE_INDICES))
                self.robotarm.apply_action(ArticulationAction(joint_positions=self.robotarm_target_position, joint_indices=UR5_INDICES))
                self.robotarm.apply_action(self.differential_controller_skid_steer(self.mobile_target_linear, self.mobile_target_angular))

                # 카메라 데이터 가져오기 (NumPy 배열)
                rgb_hand_np = self.cam_hand.get_rgb() # shape: (H, W, 4), RGBA, uint8
                rgb_mobile_np = self.cam_mobile.get_rgb() # shape: (H, W, 4), RGBA, uint8
                depth_hand_np = self.cam_hand.get_depth() # shape: (H, W), float32 (meters)
                depth_mobile_np = self.cam_mobile.get_depth() # shape: (H, W), float32 (meters)

                # RGB 이미지는 RGBA에서 RGB로 변환 필요 (SHM_CONFIG의 채널 수에 맞게)
                if rgb_hand_np.shape[2] == 4 and RGB_CHANNELS == 3:
                    rgb_hand_np = rgb_hand_np[:, :, :3] 
                if rgb_mobile_np.shape[2] == 4 and RGB_CHANNELS == 3:
                    rgb_mobile_np = rgb_mobile_np[:, :, :3]
                
                # SHM에 쓰고 신호 보내기
                self.write_images_to_shm_and_signal(rgb_hand_np, rgb_mobile_np, depth_hand_np, depth_mobile_np)
                
                self.publish_slave_info() # 다른 센서 데이터 발행
                # t_pub = time.perf_counter()

                # FPS 로깅 (선택적)
                # image_processed_count += 1
                # current_time = time.perf_counter()
                # if current_time - fps_log_start_time >= log_interval:
                #     fps = image_processed_count / (current_time - fps_log_start_time)
                #     self.get_logger().info(f"Isaac Sim image processing/SHM write FPS: {fps:.2f}")
                #     image_processed_count = 0
                #     fps_log_start_time = current_time

            # print(f"dt step={t_step-t0:.4f}s  ros={t_ros-t_step:.4f}s  "
            #       f"pub={t_pub-t_ros:.4f}s  total={t_pub-t0:.4f}s")

        # 시뮬레이션 종료 처리
        self.get_logger().info("SimulationApp not running or rclpy not ok. Exiting run loop.")
        self.timeline.stop()
        # self._cleanup_shared_memory() # atexit으로 이미 등록됨
        # self.destroy_node() # main의 finally에서 처리

def main(args=None):
    # rclpy.init(args=args) # RobotarmController 내부에서 init하지 않도록 수정 필요 또는 컨텍스트 공유
    # -> RobotarmController 생성자에서 super().__init__ 전에 rclpy.init()을 호출하면 안 됨.
    #    main에서 한 번만 init하고 노드 생성자에 컨텍스트를 전달하거나, 노드 생성 시 자동으로 처리되도록 함.
    #    가장 간단한 방법은 main에서 init하고, 노드는 그냥 생성.
    rclpy.init(args=args)
    robotarm_controller_node = None
    try:
        robotarm_controller_node = RobotarmController()
        robotarm_controller_node.run() # run 메서드 내에서 spin_once 사용
    except KeyboardInterrupt:
        print("KeyboardInterrupt, shutting down.")
    except Exception as e:
        if robotarm_controller_node:
            robotarm_controller_node.get_logger().fatal(f"Unhandled exception in main: {e}", exc_info=True)
        else:
            print(f"Unhandled exception before node creation: {e}")
    finally:
        if robotarm_controller_node:
            robotarm_controller_node.get_logger().info("Destroying node...")
            # SHM 정리는 atexit으로 하지만, 명시적 호출도 고려 가능
            robotarm_controller_node._cleanup_shared_memory() # 명시적 호출 추가
            robotarm_controller_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("RCLPY shutdown.")
        if simulation_app.is_running(): # 앱이 여전히 실행 중이면 종료
            simulation_app.close()
        print("Isaac Sim application closed.")

if __name__ == "__main__":
    main()