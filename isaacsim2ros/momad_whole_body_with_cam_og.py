"""
momad_whole_body_with_cam_bridged.py
Isaac Sim 4.5.0 + ROS 2 (Humble) — RGB/Depth × 2 토픽을 C++ Bridge로 퍼블리시
"""

# ───────────────────────────────────────────────────────────────
# 0.  기본 초기화 + Bridge Extension 미리 켜두기
# ───────────────────────────────────────────────────────────────
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": False,                         # GUI 필요 없으면 True
    "exts": ["isaacsim.ros2.bridge"],          # ★ Bridge 자동 로드 ★
})
simulation_app.set_setting("/rtx/pathtracing/enable", False)
simulation_app.set_setting("/rtx/post/dlss/enable",   True)


import omni.graph.core as og
import omni.usd
from pxr import Sdf


from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import Articulation
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid

# ───────── ROS 2 API ─────────
import rclpy, time, cv2, numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from momad_msgs.msg import ControlValue

# ──────────────────────────────────────────────────────────────
# 1.  상수 정의
# ───────────────────────────────────────────────────────────────
MAX_GRIPPER_POS = 0.025
WHEEL_RADIUS = 0.147
WHEEL_BASE   = 0.645

UR5_INDICES    = [4, 2, 0, 1, 3, 5]
HANDE_INDICES  = [6, 7]
JACKAL_INDICES = [8, 9, 10, 11]
WHOLE_INDICES  = UR5_INDICES + HANDE_INDICES + JACKAL_INDICES

# ───────────────────────────────────────────────────────────────
# 2.  ROS 2 노드
# ───────────────────────────────────────────────────────────────
class RobotarmController(Node):
    # Remove the graph tracking list since we won't use OmniGraph
    
    def __init__(self):
        super().__init__("robotarm_controller")

        # ROS 인터페이스
        self.subscription = self.create_subscription(
            ControlValue, "/master_info", self.command_callback, 1
        )
        self.publisher    = self.create_publisher(ControlValue, "/slave_info", 1)
        self.publisher_ft = self.create_publisher(Float32MultiArray, "/force", 1)

        # ── World & 자산 로드 ───────────────────────────────
        self.timeline = omni.timeline.get_timeline_interface()
        self.world = World(stage_units_in_meters=1.0,
                           physics_dt=1/240, rendering_dt=1/30)
        self.world.scene.add_default_ground_plane()

        add_reference_to_stage("/home/choiyj/Desktop/momad_test3_cam.usd", "/World")

        # ── 로봇 & 카메라 객체 ──────────────────────────────
        self.robotarm = SingleManipulator(
            prim_path="/World/ur5",
            end_effector_prim_path="/World/ur5/wrist_3_link/flange",
            name="ur5",
        )
        self.gripper = Articulation("/World/hande")

        # Configure cameras with proper annotators for both RGB and depth
        self.cam_mobile = Camera(
            prim_path="/World/jackal_basic/base_link/RSD455/Camera_Pseudo_Depth",
            frequency=30, 
            resolution=(1280, 720),
        )
        self.cam_hand = Camera(
            prim_path="/World/hande/tool0/RSD455/Camera_Pseudo_Depth",
            frequency=30, 
            resolution=(1280, 720),
        )

        # 초기화
        self.world.reset()
        self.robotarm.initialize()
        self.gripper.initialize()
        self.cam_mobile.initialize()
        self.cam_hand.initialize()

        self.cam_hand.add_distance_to_image_plane_to_frame()
        self.cam_mobile.add_distance_to_image_plane_to_frame()

        # 제어용 변수
        self.robotarm_target_position  = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
        self.robotarm_target_velocity  = [0.0] * 6
        self.gripper_target_position   = [0.0] * 2
        self.gripper_target_velocity   = [0.0] * 2
        self.mobile_target_linear      = 0.0
        self.mobile_target_angular     = 0.0

        # 안정화
        self.robotarm.set_joint_positions(self.robotarm_target_position, UR5_INDICES)
        for _ in range(40):
            self.world.step(render=True)

        # 테스트 큐브
        self.world.scene.add(
            DynamicCuboid(
                name="cube1", prim_path="/World/Cube1",
                position=np.array([0.9, -0.1, 0.9]),
                scale=np.array([0.03, 0.03, 0.03]), size=1.0,
                color=np.array([0, 1, 0]),
            )
        )
        # 테스트 큐브
        self.world.scene.add(
            DynamicCuboid(
                name="cube2",
                position=np.array([0.9, 0.0, 0.4]),
                prim_path="/World/Cube2",
                scale=np.array([0.5, 0.5, 0.8]),
                size=1.0,
                color=np.array([0.05, 0.1, 0.1]),
            )
        )

        # Camera publishers
        from sensor_msgs.msg import CompressedImage
        self.mobile_rgb_pub = self.create_publisher(CompressedImage, "/mobile_cam/rgb", 10)
        self.mobile_depth_pub = self.create_publisher(CompressedImage, "/mobile_cam/depth", 10)
        self.hand_rgb_pub = self.create_publisher(CompressedImage, "/hand_cam/rgb", 10) 
        self.hand_depth_pub = self.create_publisher(CompressedImage, "/hand_cam/depth", 10)
        
        self.image_msg = CompressedImage()

    def _publish_camera_images(self):
        """Publish camera images without using OmniGraph"""
        try:
            # Mobile camera
            mobile_rgb = self.cam_mobile.get_rgba()
            if mobile_rgb is not None:
                # Convert to BGR for OpenCV
                mobile_rgb_bgr = cv2.cvtColor(mobile_rgb, cv2.COLOR_RGBA2BGR)
                # Compress image
                _, compressed = cv2.imencode('.jpg', mobile_rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                self.image_msg.header.stamp = self.get_clock().now().to_msg()
                self.image_msg.header.frame_id = "mobile_cam_rgb"
                self.image_msg.format = "jpeg"
                self.image_msg.data = compressed.tobytes()
                self.mobile_rgb_pub.publish(self.image_msg)
                
            # Mobile depth
            mobile_depth = self.cam_mobile.get_depth()
            if mobile_depth is not None:
                # Normalize depth for visualization
                norm_depth = np.uint8((mobile_depth / np.max(mobile_depth)) * 255)
                # Compress image
                _, compressed = cv2.imencode('.jpg', norm_depth, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                self.image_msg.header.stamp = self.get_clock().now().to_msg()
                self.image_msg.header.frame_id = "mobile_cam_depth"
                self.image_msg.format = "jpeg"
                self.image_msg.data = compressed.tobytes()
                self.mobile_depth_pub.publish(self.image_msg)
                
            # Hand camera
            hand_rgb = self.cam_hand.get_rgba()
            if hand_rgb is not None:
                # Convert to BGR for OpenCV
                hand_rgb_bgr = cv2.cvtColor(hand_rgb, cv2.COLOR_RGBA2BGR)
                # Compress image
                _, compressed = cv2.imencode('.jpg', hand_rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                self.image_msg.header.stamp = self.get_clock().now().to_msg()
                self.image_msg.header.frame_id = "hand_cam_rgb"
                self.image_msg.format = "jpeg"
                self.image_msg.data = compressed.tobytes()
                self.hand_rgb_pub.publish(self.image_msg)
                
            # Hand depth
            hand_depth = self.cam_hand.get_depth()
            if hand_depth is not None:
                # Normalize depth for visualization
                norm_depth = np.uint8((hand_depth / np.max(hand_depth)) * 255)
                # Compress image
                _, compressed = cv2.imencode('.jpg', norm_depth, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                self.image_msg.header.stamp = self.get_clock().now().to_msg()
                self.image_msg.header.frame_id = "hand_cam_depth"
                self.image_msg.format = "jpeg"
                self.image_msg.data = compressed.tobytes()
                self.hand_depth_pub.publish(self.image_msg)
                
        except Exception as e:
            print(f"Error publishing camera images: {e}")

    # ────────────────────── ROS Command 콜백 ───────────────────
    def command_callback(self, msg):
        # UR5
        self.robotarm_target_position = [
            np.clip(a * np.pi / 180.0, -3.14, 3.14) for a in msg.robotarm_state.position
        ]
        self.robotarm_target_velocity = msg.robotarm_state.velocity

        # Hand-E
        self.gripper_target_position = [
            np.clip(a, 0.0, 1.0) * MAX_GRIPPER_POS for a in msg.gripper_state.position
        ]
        self.gripper_target_velocity = msg.gripper_state.velocity

        # Jackal
        self.mobile_target_linear  = msg.mobile_state.linear_velocity
        self.mobile_target_angular = msg.mobile_state.angular_velocity

    # ────────────────────── 상태 퍼블리시 ──────────────────────
    def publish_slave_info(self):
        msg = ControlValue()
        pos, vel, force, lin_v, ang_v, ft = self.get_robot_state()

        msg.robotarm_state.position = pos[0:6]
        msg.robotarm_state.velocity = vel[0:6]
        msg.robotarm_state.force    = force[0:6]

        msg.gripper_state.position = [np.clip(p / MAX_GRIPPER_POS, 0.0, 1.0)
                                      for p in pos[6:8]]
        msg.gripper_state.velocity = vel[6:8]
        msg.gripper_state.force    = force[6:8]

        msg.mobile_state.linear_velocity  = float(lin_v)
        msg.mobile_state.angular_velocity = float(ang_v)
        msg.force_torque = ft[6]          # 예시

        msg.stamp = time.time()
        self.publisher.publish(msg)

    # ────────────────────── 로봇 상태 측정 ────────────────────
    def get_robot_state(self):
        pos  = self.robotarm.get_joint_positions(joint_indices=WHOLE_INDICES).tolist()
        vel  = self.robotarm.get_joint_velocities(joint_indices=WHOLE_INDICES).tolist()
        tau  = self.robotarm.get_measured_joint_efforts(joint_indices=WHOLE_INDICES).tolist()
        g    = self.gripper.get_generalized_gravity_forces(joint_indices=WHOLE_INDICES)[0]
        c    = self.gripper.get_coriolis_and_centrifugal_forces(joint_indices=WHOLE_INDICES)[0]
        real_tau = [t - gg - cc for t, gg, cc in zip(tau, g, c)]
        ft = self.robotarm.get_measured_joint_forces(joint_indices=WHOLE_INDICES).tolist()

        w_L = np.mean([vel[JACKAL_INDICES[0]], vel[JACKAL_INDICES[2]]])
        w_R = np.mean([vel[JACKAL_INDICES[1]], vel[JACKAL_INDICES[3]]])
        lin_v = WHEEL_RADIUS * 0.5 * (w_L + w_R)
        ang_v = WHEEL_RADIUS / WHEEL_BASE * (w_R - w_L)

        return pos, vel, real_tau, lin_v, ang_v, ft

    # ────────────────────── Jackal 제어 ──────────────────────
    def differential_controller(self, v_lin, v_ang):
        chi = 1.4
        eff_L = ((2*v_lin) - v_ang * (WHEEL_BASE*chi)) / (2*WHEEL_RADIUS)
        eff_R = ((2*v_lin) + v_ang * (WHEEL_BASE*chi)) / (2*WHEEL_RADIUS)

        fl = eff_L - 0.08*v_ang
        rl = fl - 0.03*v_ang
        fr = eff_R + 0.00*v_ang
        rr = fr + 0.06*v_ang

        max_w = 1000.0
        speeds = np.clip([fl, fr, rl, rr], -max_w, max_w).tolist()
        return ArticulationAction(joint_velocities=speeds, joint_indices=JACKAL_INDICES)

    # ────────────────────── 메인 루프 ────────────────────────
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
                    self.world.reset(); reset_needed = False

                # UR5 & Hand-E
                self.robotarm.apply_action(ArticulationAction(
                    joint_positions=self.gripper_target_position,
                    joint_velocities=self.gripper_target_velocity,
                    joint_indices=HANDE_INDICES))
                self.robotarm.apply_action(ArticulationAction(
                    joint_positions=self.robotarm_target_position,
                    joint_indices=UR5_INDICES))

                # Jackal
                self.robotarm.apply_action(
                    self.differential_controller(self.mobile_target_linear,
                                                 self.mobile_target_angular))

                # Publish to ROS topics
                self.publish_slave_info()
                # Use our direct camera publishing method
                self._publish_camera_images()
                t_pub = time.perf_counter()

                print(f"dt step={t_step-t0:.4f}s  ros={t_ros-t_step:.4f}s  "
                      f"pub={t_pub-t_ros:.4f}s  total={t_pub-t0:.4f}s")

        self.timeline.stop()
        self.destroy_node()
        simulation_app.close()

# ───────────────────────────────────────────────────────────────
# 3.  실행
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rclpy.init()
    node = RobotarmController()
    node.run()
