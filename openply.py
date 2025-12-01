"""
PLY Point Cloud Visualization Tool
PLY点云可视化工具 - 用于查看探索结果

使用Open3D库实现交互式点云查看功能
"""

import argparse
import os
import numpy as np
import open3d as o3d
from typing import List, Optional


def load_ply_file(filepath: str) -> o3d.geometry.PointCloud:
    """
    加载PLY文件

    Args:
        filepath: PLY文件路径

    Returns:
        point_cloud: Open3D点云对象
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PLY file not found: {filepath}")

    print(f"Loading {filepath}...")
    pcd = o3d.io.read_point_cloud(filepath)
    print(f"  Loaded {len(pcd.points)} points")

    # 如果没有颜色，使用默认颜色
    if not pcd.has_colors():
        print("  No color information, using default gray")
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    return pcd


def create_coordinate_frame(size: float = 1.0, origin: np.ndarray = None) -> o3d.geometry.TriangleMesh:
    """
    创建坐标系（用于可视化参考）

    Args:
        size: 坐标轴长度
        origin: 坐标原点位置

    Returns:
        coordinate_frame: 坐标系网格对象
    """
    if origin is None:
        origin = [0, 0, 0]

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=origin
    )
    return coord_frame


def create_trajectory_line(trajectory_ply_path: str, color: List[float] = None) -> o3d.geometry.LineSet:
    """
    从轨迹PLY文件创建线段（用于可视化轨迹）

    Args:
        trajectory_ply_path: 轨迹PLY文件路径
        color: 线段颜色 [R, G, B]，范围0-1

    Returns:
        line_set: 轨迹线段对象
    """
    if color is None:
        color = [1, 0, 0]  # 默认红色

    # 加载轨迹点
    trajectory_pcd = load_ply_file(trajectory_ply_path)
    points = np.asarray(trajectory_pcd.points)

    if len(points) < 2:
        print("Warning: Trajectory has less than 2 points, cannot create line")
        return None

    # 创建线段连接
    lines = [[i, i + 1] for i in range(len(points) - 1)]

    # 创建LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # 设置颜色
    colors = [color for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    print(f"Created trajectory line with {len(points)} waypoints")

    return line_set


def create_trajectory_spheres(trajectory_ply_path: str, radius: float = 0.05,
                              color: List[float] = None) -> List[o3d.geometry.TriangleMesh]:
    """
    从轨迹PLY文件创建球体标记（用于突出显示轨迹点）

    Args:
        trajectory_ply_path: 轨迹PLY文件路径
        radius: 球体半径
        color: 球体颜色 [R, G, B]

    Returns:
        spheres: 球体列表
    """
    if color is None:
        color = [1, 0.5, 0]  # 默认橙色

    trajectory_pcd = load_ply_file(trajectory_ply_path)
    points = np.asarray(trajectory_pcd.points)

    spheres = []
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)

    print(f"Created {len(spheres)} trajectory markers")

    return spheres


def visualize_ply_files(ply_files: List[str],
                       show_coordinate_frame: bool = True,
                       coordinate_frame_size: float = 1.0,
                       trajectory_file: Optional[str] = None,
                       show_trajectory_line: bool = True,
                       show_trajectory_markers: bool = True,
                       point_size: float = 2.0,
                       background_color: List[float] = None):
    """
    可视化一个或多个PLY文件

    Args:
        ply_files: PLY文件路径列表
        show_coordinate_frame: 是否显示坐标系
        coordinate_frame_size: 坐标系大小
        trajectory_file: 轨迹PLY文件路径（可选）
        show_trajectory_line: 是否显示轨迹连线
        show_trajectory_markers: 是否显示轨迹点标记
        point_size: 点云显示大小
        background_color: 背景颜色 [R, G, B]
    """
    if background_color is None:
        background_color = [0.1, 0.1, 0.1]  # 深灰色背景

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Point Cloud Viewer", width=1280, height=720)

    # 加载并添加点云
    geometries = []

    for ply_file in ply_files:
        pcd = load_ply_file(ply_file)
        geometries.append(pcd)
        vis.add_geometry(pcd)

    # 添加坐标系
    if show_coordinate_frame:
        coord_frame = create_coordinate_frame(size=coordinate_frame_size)
        geometries.append(coord_frame)
        vis.add_geometry(coord_frame)

    # 添加轨迹
    if trajectory_file is not None and os.path.exists(trajectory_file):
        print(f"\nVisualizing trajectory from {trajectory_file}")

        # 轨迹连线
        if show_trajectory_line:
            line_set = create_trajectory_line(trajectory_file, color=[1, 0, 0])
            if line_set is not None:
                geometries.append(line_set)
                vis.add_geometry(line_set)

        # 轨迹点标记
        if show_trajectory_markers:
            spheres = create_trajectory_spheres(trajectory_file, radius=0.05, color=[1, 0.5, 0])
            for sphere in spheres:
                geometries.append(sphere)
                vis.add_geometry(sphere)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array(background_color)

    # 设置视角
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)

    print("\n" + "="*60)
    print("Visualization Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Shift + Mouse drag: Pan")
    print("  - Press 'H': Show help")
    print("  - Press 'Q' or ESC: Exit")
    print("="*60 + "\n")

    # 运行可视化
    vis.run()
    vis.destroy_window()


def compare_ply_files(ply_files: List[str], colors: List[List[float]] = None):
    """
    比较多个PLY文件（使用不同颜色）

    Args:
        ply_files: PLY文件路径列表
        colors: 每个点云的颜色列表
    """
    if colors is None:
        # 默认颜色
        colors = [
            [1, 0, 0],    # 红色
            [0, 1, 0],    # 绿色
            [0, 0, 1],    # 蓝色
            [1, 1, 0],    # 黄色
            [1, 0, 1],    # 品红色
            [0, 1, 1],    # 青色
        ]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Comparison Viewer", width=1280, height=720)

    for i, ply_file in enumerate(ply_files):
        pcd = load_ply_file(ply_file)

        # 使用指定颜色
        if i < len(colors):
            pcd.paint_uniform_color(colors[i])

        vis.add_geometry(pcd)

    # 添加坐标系
    coord_frame = create_coordinate_frame(size=1.0)
    vis.add_geometry(coord_frame)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])

    print("\nComparing multiple point clouds:")
    for i, ply_file in enumerate(ply_files):
        color_str = f"RGB{tuple(colors[i % len(colors)])}" if i < len(colors) else "default"
        print(f"  [{color_str}] {os.path.basename(ply_file)}")

    vis.run()
    vis.destroy_window()


def get_point_cloud_info(ply_file: str):
    """
    获取并打印点云信息

    Args:
        ply_file: PLY文件路径
    """
    pcd = load_ply_file(ply_file)
    points = np.asarray(pcd.points)

    print(f"\n{'='*60}")
    print(f"Point Cloud Information: {os.path.basename(ply_file)}")
    print(f"{'='*60}")
    print(f"Number of points: {len(points)}")
    print(f"Has colors: {pcd.has_colors()}")
    print(f"Has normals: {pcd.has_normals()}")

    if len(points) > 0:
        print(f"\nBounding box:")
        print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

        center = points.mean(axis=0)
        print(f"\nCenter: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PLY Point Cloud Visualization Tool')
    parser.add_argument('files', nargs='+', help='PLY file(s) to visualize')
    parser.add_argument('--trajectory', type=str, default=None,
                       help='Trajectory PLY file (optional)')
    parser.add_argument('--no-coord', action='store_true',
                       help='Do not show coordinate frame')
    parser.add_argument('--coord-size', type=float, default=1.0,
                       help='Coordinate frame size (default: 1.0)')
    parser.add_argument('--no-trajectory-line', action='store_true',
                       help='Do not show trajectory line')
    parser.add_argument('--no-trajectory-markers', action='store_true',
                       help='Do not show trajectory markers')
    parser.add_argument('--point-size', type=float, default=2.0,
                       help='Point size (default: 2.0)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare mode: show multiple files with different colors')
    parser.add_argument('--info', action='store_true',
                       help='Show point cloud information only (no visualization)')

    args = parser.parse_args()

    # 检查文件是否存在
    for f in args.files:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            exit(1)

    # 仅显示信息
    if args.info:
        for ply_file in args.files:
            get_point_cloud_info(ply_file)
        exit(0)

    # 比较模式
    if args.compare:
        compare_ply_files(args.files)
    else:
        # 正常可视化
        visualize_ply_files(
            ply_files=args.files,
            show_coordinate_frame=not args.no_coord,
            coordinate_frame_size=args.coord_size,
            trajectory_file=args.trajectory,
            show_trajectory_line=not args.no_trajectory_line,
            show_trajectory_markers=not args.no_trajectory_markers,
            point_size=args.point_size
        )

    print("Visualization closed.")
