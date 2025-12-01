"""
RealSense D435i Camera Implementation
实现RealSense D435i相机接口，用于实体机器人部署
"""

import numpy as np
import torch
import pyrealsense2 as rs
import cv2
from typing import Tuple, Optional


class RealSenseCamera:
    """
    RealSense D435i相机封装类
    提供与仿真相机类似的接口，用于实体机器人部署
    """

    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 device: str = 'cuda:0',
                 enable_align: bool = True):
        """
        初始化RealSense D435i相机

        Args:
            width: 图像宽度 (默认640)
            height: 图像高度 (默认480)
            fps: 帧率 (默认30)
            device: PyTorch设备 ('cuda:0' 或 'cpu')
            enable_align: 是否对齐深度图到RGB图 (默认True)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.device = torch.device(device)
        self.enable_align = enable_align

        # 初始化标志（用于安全清理）
        self.pipeline = None
        self.profile = None

        try:
            # 创建pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # 配置流
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            # 启动pipeline
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error: Failed to initialize RealSense camera: {e}")
            print("Please check:")
            print("  1. Camera is connected via USB")
            print("  2. RealSense SDK is installed correctly")
            print("  3. No other program is using the camera")
            raise RuntimeError(f"RealSense camera initialization failed: {e}")

        # 获取相机内参
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # 获取内参矩阵
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)

        self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # 创建对齐对象（将深度图对齐到RGB图）
        if self.enable_align:
            self.align = rs.align(rs.stream.color)

        # 创建滤波器以提高深度图质量
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

        # 预热相机（丢弃前几帧）
        print("Warming up RealSense camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("RealSense camera ready!")

    def get_intrinsics(self):
        """
        获取相机内参

        Returns:
            Dict: 包含fx, fy, cx, cy的字典
        """
        return {
            'fx': self.color_intrinsics.fx,
            'fy': self.color_intrinsics.fy,
            'cx': self.color_intrinsics.ppx,
            'cy': self.color_intrinsics.ppy,
            'width': self.color_intrinsics.width,
            'height': self.color_intrinsics.height
        }

    def capture_image(self,
                     apply_filters: bool = True,
                     max_depth: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        捕获RGB图像和深度图

        Args:
            apply_filters: 是否应用深度滤波器 (默认True)
            max_depth: 最大深度值（米），超过此值的深度设为无效 (默认10.0m)

        Returns:
            images: RGB图像 Tensor, shape (1, H, W, 3), 范围[0, 1]
            depth: 深度图 Tensor, shape (1, H, W, 1), 单位为米
        """
        # 等待一帧数据
        frames = self.pipeline.wait_for_frames()

        # 对齐深度图到RGB图
        if self.enable_align:
            frames = self.align.process(frames)

        # 获取深度帧和彩色帧
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to capture frames from RealSense camera")

        # 应用深度滤波器
        if apply_filters:
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.hole_filling_filter.process(depth_frame)

        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 深度图转换为米
        depth_in_meters = depth_image * self.depth_scale

        # 标记无效深度（原始为0的像素）
        invalid_mask = (depth_image == 0)

        # 限制最大深度（超过max_depth的设为无效）
        depth_in_meters[depth_in_meters > max_depth] = 0.0

        # 将所有无效深度统一设为0（与实机模式一致）
        depth_in_meters[invalid_mask] = 0.0

        # BGR转RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 归一化RGB图像到[0, 1]
        color_image = color_image.astype(np.float32) / 255.0

        # 转换为PyTorch张量
        images = torch.from_numpy(color_image).unsqueeze(0).to(self.device)  # (1, H, W, 3)
        depth = torch.from_numpy(depth_in_meters).unsqueeze(0).unsqueeze(-1).to(self.device)  # (1, H, W, 1)

        return images, depth

    def capture_raw_frames(self):
        """
        捕获原始帧（不进行处理）

        Returns:
            depth_frame: RealSense深度帧对象
            color_frame: RealSense彩色帧对象
        """
        frames = self.pipeline.wait_for_frames()

        if self.enable_align:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        return depth_frame, color_frame

    def get_point_cloud(self, depth_frame, color_frame=None):
        """
        从深度帧生成点云

        Args:
            depth_frame: RealSense深度帧
            color_frame: RealSense彩色帧（可选）

        Returns:
            points: Nx3点云坐标
            colors: Nx3点云颜色（如果提供了color_frame）
        """
        pc = rs.pointcloud()

        if color_frame is not None:
            pc.map_to(color_frame)

        points = pc.calculate(depth_frame)

        # 获取顶点坐标
        vertices = np.asanyarray(points.get_vertices())
        points_array = np.array([[v[0], v[1], v[2]] for v in vertices])

        if color_frame is not None:
            # 获取颜色
            texture_coords = np.asanyarray(points.get_texture_coordinates())
            color_image = np.asanyarray(color_frame.get_data())

            # BGR转RGB
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            colors = []
            for tc in texture_coords:
                x = int(tc[0] * color_image.shape[1])
                y = int(tc[1] * color_image.shape[0])
                if 0 <= x < color_image.shape[1] and 0 <= y < color_image.shape[0]:
                    colors.append(color_image[y, x] / 255.0)
                else:
                    colors.append([0, 0, 0])
            colors_array = np.array(colors)

            return points_array, colors_array

        return points_array, None

    def compute_partial_point_cloud(self,
                                    depth: torch.Tensor,
                                    images: Optional[torch.Tensor] = None,
                                    gathering_factor: float = 0.05) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        从深度图计算局部点云（与仿真Camera类兼容）

        Args:
            depth: 深度图 Tensor, shape (1, H, W, 1)
            images: RGB图像 Tensor, shape (1, H, W, 3) (可选)
            gathering_factor: 采样比例 (默认0.05)

        Returns:
            world_points: 点云坐标 (N, 3)
            points_color: 点云颜色 (N, 3) (如果提供了images)
        """
        # 获取内参
        fx = self.color_intrinsics.fx
        fy = self.color_intrinsics.fy
        cx = self.color_intrinsics.ppx
        cy = self.color_intrinsics.ppy

        # 深度图转numpy
        depth_np = depth.squeeze().cpu().numpy()  # (H, W)

        # 创建像素坐标网格
        h, w = depth_np.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # 有效深度掩码
        valid_mask = depth_np > 0

        # 反投影到3D
        z = depth_np[valid_mask]
        x = (u[valid_mask] - cx) * z / fx
        y = (v[valid_mask] - cy) * z / fy

        # 组合成点云（相机坐标系）
        points_camera = np.stack([x, y, z], axis=-1)  # (N, 3)

        # 转换为PyTorch张量
        world_points = torch.from_numpy(points_camera).float().to(self.device)

        # 采样
        n_points = int(len(world_points) * gathering_factor)
        if n_points > 0 and n_points < len(world_points):
            indices = torch.randperm(len(world_points))[:n_points]
            world_points = world_points[indices]

        # 提取颜色
        points_color = None
        if images is not None:
            images_np = images.squeeze().cpu().numpy()  # (H, W, 3)
            colors = images_np[valid_mask]  # (N, 3)
            points_color = torch.from_numpy(colors).float().to(self.device)

            if n_points > 0 and n_points < len(points_color):
                points_color = points_color[indices]

        return world_points, points_color

    def stop(self):
        """停止相机"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                print("RealSense camera stopped")
            except Exception as e:
                print(f"Warning: Error stopping camera: {e}")

    def __del__(self):
        """析构函数，确保相机正确关闭"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except:
                pass


def test_realsense():
    """测试RealSense相机"""
    print("Testing RealSense D435i camera...")

    # 初始化相机
    camera = RealSenseCamera(width=640, height=480, device='cpu')

    # 获取内参
    intrinsics = camera.get_intrinsics()
    print(f"Camera intrinsics: {intrinsics}")

    # 捕获图像
    try:
        for i in range(10):
            images, depth = camera.capture_image()
            print(f"Frame {i}: RGB shape={images.shape}, Depth shape={depth.shape}")
            print(f"  Depth range: [{depth[depth > 0].min():.3f}, {depth[depth > 0].max():.3f}] meters")

            # 计算点云
            world_points, colors = camera.compute_partial_point_cloud(depth, images)
            print(f"  Point cloud: {len(world_points)} points")

    finally:
        camera.stop()

    print("Test completed!")


if __name__ == "__main__":
    test_realsense()
