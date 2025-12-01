"""
Real World Environment for Robot Deployment
实体机器人环境管理，整合RealSense相机和Unitree Go2w机器狗
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
from scipy.spatial.transform import Rotation as R
from real_camera_impl import RealSenseCamera


class RobotPose:
    """机器人位姿类，用于存储和转换位姿信息"""

    def __init__(self, position: np.ndarray, orientation: np.ndarray, frame: str = 'robot'):
        """
        初始化机器人位姿

        Args:
            position: 位置 (x, y, z) in meters
            orientation: 方向，可以是四元数(w, x, y, z)或欧拉角(roll, pitch, yaw)
            frame: 坐标系 ('robot', 'world', 'camera')
        """
        self.position = np.array(position, dtype=np.float32)
        self.orientation = np.array(orientation, dtype=np.float32)
        self.frame = frame

        # 如果是欧拉角（3个值），转换为四元数
        if len(self.orientation) == 3:
            r = R.from_euler('xyz', self.orientation)
            self.quaternion = r.as_quat()  # (x, y, z, w)
        else:
            self.quaternion = self.orientation

    def get_transformation_matrix(self) -> np.ndarray:
        """
        获取4x4变换矩阵

        Returns:
            T: 4x4 transformation matrix
        """
        # 四元数转旋转矩阵
        r = R.from_quat(self.quaternion)
        rotation_matrix = r.as_matrix()

        # 构建4x4变换矩阵
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = self.position

        return T

    def get_rotation_matrix(self) -> np.ndarray:
        """获取3x3旋转矩阵"""
        r = R.from_quat(self.quaternion)
        return r.as_matrix()

    def get_euler_angles(self) -> np.ndarray:
        """获取欧拉角 (roll, pitch, yaw)"""
        r = R.from_quat(self.quaternion)
        return r.as_euler('xyz')

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        euler = self.get_euler_angles()
        return {
            'position': self.position.tolist(),
            'quaternion': self.quaternion.tolist(),
            'euler': euler.tolist(),
            'frame': self.frame
        }


class RealWorldEnvironment:
    """
    实体机器人环境类
    整合RealSense D435i相机和Unitree Go2w机器狗，提供统一接口
    """

    def __init__(self,
                 camera_width: int = 640,
                 camera_height: int = 480,
                 camera_fps: int = 30,
                 device: str = 'cuda:0',
                 camera_height_offset: float = 0.3,  # 相机相对机器人底盘的高度(米)
                 camera_forward_offset: float = 0.2):  # 相机相对机器人中心的前向偏移(米)
        """
        初始化实体机器人环境

        Args:
            camera_width: 相机图像宽度
            camera_height: 相机图像高度
            camera_fps: 相机帧率
            device: PyTorch设备
            camera_height_offset: 相机相对机器人底盘的高度偏移(米)
            camera_forward_offset: 相机相对机器人中心的前向偏移(米)
        """
        self.device = torch.device(device)

        # 初始化RealSense相机
        print("Initializing RealSense D435i camera...")
        self.camera = RealSenseCamera(
            width=camera_width,
            height=camera_height,
            fps=camera_fps,
            device=device
        )

        # 相机到机器人的变换参数
        self.camera_height_offset = camera_height_offset
        self.camera_forward_offset = camera_forward_offset

        # 机器人当前位姿（世界坐标系）
        self.robot_pose = RobotPose(
            position=[0.0, 0.0, 0.0],
            orientation=[0.0, 0.0, 0.0],  # (roll, pitch, yaw)
            frame='world'
        )

        # 位姿历史
        self.pose_history = []

        # 世界坐标系原点（初始机器人位置）
        self.world_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        print("Real world environment initialized!")

    def update_robot_pose_from_state(self, state):
        """
        从机器狗状态更新机器人位姿

        Args:
            state: SportModeState_ 对象，包含机器人状态信息

        Note:
            根据Unitree Go2 SDK文档，SportModeState_对象包含以下字段：
            - position: 位置 [x, y, z]
            - imu_state: IMU状态
              - rpy: 欧拉角 [roll, pitch, yaw]
            如果字段名不匹配，请根据实际SDK调整
        """
        if state is None:
            return

        # 从state中提取位姿信息
        try:
            # 位置 (相对于初始位置)
            # 注意：Unitree SDK可能使用不同的字段名，请根据实际情况调整
            if hasattr(state, 'position'):
                # 检查position是list还是有x,y,z属性
                if isinstance(state.position, (list, tuple)) and len(state.position) >= 3:
                    position = np.array([
                        float(state.position[0]),
                        float(state.position[1]),
                        float(state.position[2])
                    ], dtype=np.float32)
                elif hasattr(state.position, 'x'):
                    position = np.array([
                        float(state.position.x),
                        float(state.position.y),
                        float(state.position.z)
                    ], dtype=np.float32)
                else:
                    position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # 姿态 (IMU数据)
            # 注意：需要根据实际SDK字段调整
            if hasattr(state, 'imu_state') and hasattr(state.imu_state, 'rpy'):
                rpy = state.imu_state.rpy
                if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
                    roll = float(rpy[0])
                    pitch = float(rpy[1])
                    yaw = float(rpy[2])
                else:
                    roll, pitch, yaw = 0.0, 0.0, 0.0
            else:
                roll, pitch, yaw = 0.0, 0.0, 0.0

            orientation = np.array([roll, pitch, yaw], dtype=np.float32)

            # 更新机器人位姿
            self.robot_pose = RobotPose(
                position=position + self.world_origin,
                orientation=orientation,
                frame='world'
            )

        except Exception as e:
            print(f"Warning: Failed to update robot pose from state: {e}")
            print(f"  State type: {type(state)}")
            print(f"  Available attributes: {dir(state) if hasattr(state, '__dir__') else 'N/A'}")

    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相机在世界坐标系下的位姿

        Returns:
            R: 3x3旋转矩阵
            T: 3x1平移向量
        """
        # 获取机器人位姿的变换矩阵
        robot_T = self.robot_pose.get_transformation_matrix()

        # 相机相对机器人的偏移（机器人坐标系）
        camera_offset_robot = np.array([
            self.camera_forward_offset,  # x: 向前
            0.0,                          # y: 左右
            self.camera_height_offset     # z: 向上
        ], dtype=np.float32)

        # 将相机偏移转换到世界坐标系
        camera_offset_world = robot_T[:3, :3] @ camera_offset_robot

        # 相机在世界坐标系下的位置
        camera_position = self.robot_pose.position + camera_offset_world

        # 相机姿态与机器人相同（假设相机朝向与机器人一致）
        camera_rotation = self.robot_pose.get_rotation_matrix()

        # 转换为PyTorch3D格式 (R, T)
        R_torch = camera_rotation
        T_torch = camera_position

        return R_torch, T_torch

    def get_camera_pose_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取相机位姿的PyTorch张量格式

        Returns:
            R: (1, 3, 3) 旋转矩阵
            T: (1, 3) 平移向量
        """
        R_np, T_np = self.get_camera_pose()

        R_torch = torch.from_numpy(R_np).unsqueeze(0).float().to(self.device)
        T_torch = torch.from_numpy(T_np).unsqueeze(0).float().to(self.device)

        return R_torch, T_torch

    def capture_image_with_pose(self, apply_filters: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        捕获图像和深度，同时返回相机位姿

        Args:
            apply_filters: 是否应用深度滤波器

        Returns:
            images: RGB图像 (1, H, W, 3)
            depth: 深度图 (1, H, W, 1)
            R: 旋转矩阵 (1, 3, 3)
            T: 平移向量 (1, 3)
        """
        # 捕获图像
        images, depth = self.camera.capture_image(apply_filters=apply_filters)

        # 获取相机位姿
        R, T = self.get_camera_pose_torch()

        return images, depth, R, T

    def compute_partial_point_cloud_world(self,
                                         depth: torch.Tensor,
                                         images: Optional[torch.Tensor] = None,
                                         R: Optional[torch.Tensor] = None,
                                         T: Optional[torch.Tensor] = None,
                                         gathering_factor: float = 0.05) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算世界坐标系下的局部点云

        Args:
            depth: 深度图 (1, H, W, 1)
            images: RGB图像 (1, H, W, 3)
            R: 旋转矩阵 (1, 3, 3)，如果为None则使用当前位姿
            T: 平移向量 (1, 3)，如果为None则使用当前位姿
            gathering_factor: 采样比例

        Returns:
            world_points: 世界坐标系下的点云 (N, 3)
            points_color: 点云颜色 (N, 3)
        """
        # 计算相机坐标系下的点云
        camera_points, points_color = self.camera.compute_partial_point_cloud(
            depth, images, gathering_factor
        )

        # 获取位姿
        if R is None or T is None:
            R, T = self.get_camera_pose_torch()

        # 转换到世界坐标系
        # world_points = R @ camera_points^T + T
        R_matrix = R.squeeze(0)  # (3, 3)
        T_vector = T.squeeze(0)  # (3,)

        world_points = (R_matrix @ camera_points.T).T + T_vector  # (N, 3)

        return world_points, points_color

    def save_current_pose(self):
        """保存当前位姿到历史记录"""
        pose_dict = self.robot_pose.to_dict()
        self.pose_history.append(pose_dict)

    def get_pose_history(self) -> list:
        """获取位姿历史"""
        return self.pose_history

    def reset_world_origin(self):
        """
        将当前位置设为世界坐标系原点
        用于初始化或重新校准
        """
        self.world_origin = self.robot_pose.position.copy()
        self.robot_pose.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pose_history = []
        print("World origin reset to current robot position")

    def stop(self):
        """停止环境（关闭相机）"""
        if self.camera is not None:
            try:
                self.camera.stop()
                print("Real world environment stopped")
            except Exception as e:
                print(f"Warning: Error stopping environment: {e}")

    def __del__(self):
        """析构函数"""
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass


def test_real_world_env():
    """测试实体机器人环境"""
    print("Testing Real World Environment...")

    # 初始化环境
    env = RealWorldEnvironment(device='cpu')

    try:
        # 捕获图像和位姿
        for i in range(5):
            images, depth, R, T = env.capture_image_with_pose()

            print(f"\nFrame {i}:")
            print(f"  Image shape: {images.shape}")
            print(f"  Depth shape: {depth.shape}")
            print(f"  Camera position: {T.squeeze().cpu().numpy()}")
            print(f"  Camera rotation shape: {R.shape}")

            # 计算世界坐标系点云
            world_points, colors = env.compute_partial_point_cloud_world(depth, images)
            print(f"  Point cloud: {len(world_points)} points in world frame")

            # 保存位姿
            env.save_current_pose()

        # 打印位姿历史
        print(f"\nPose history: {len(env.get_pose_history())} poses saved")

    finally:
        env.stop()

    print("Test completed!")


if __name__ == "__main__":
    test_real_world_env()
