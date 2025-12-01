"""
RViz Trajectory Visualization
RViz轨迹可视化工具 - 用于在RViz中显示机器人探索轨迹

需要ROS环境支持
"""

import argparse
import json
import os
import numpy as np
import time

try:
    import rospy
    from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray
    from visualization_msgs.msg import Marker, MarkerArray
    from nav_msgs.msg import Path
    from std_msgs.msg import Header, ColorRGBA
    ROS_AVAILABLE = True
except ImportError:
    print("Warning: ROS not available. Install ROS and rospy to use RViz visualization.")
    ROS_AVAILABLE = False


class RVizTrajectoryVisualizer:
    """RViz轨迹可视化器"""

    def __init__(self, frame_id='map'):
        """
        初始化可视化器

        Args:
            frame_id: TF frame ID (默认'map')
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available. Cannot initialize RViz visualizer.")

        self.frame_id = frame_id

        # 初始化ROS节点
        rospy.init_node('nbp_trajectory_visualizer', anonymous=True)

        # 创建发布器
        self.path_pub = rospy.Publisher('/nbp/trajectory_path', Path, queue_size=10)
        self.poses_pub = rospy.Publisher('/nbp/trajectory_poses', PoseArray, queue_size=10)
        self.markers_pub = rospy.Publisher('/nbp/trajectory_markers', MarkerArray, queue_size=10)
        self.point_cloud_pub = rospy.Publisher('/nbp/point_cloud_markers', Marker, queue_size=10)

        print("RViz visualizer initialized")
        print(f"  Frame ID: {frame_id}")
        print("  Publishing to topics:")
        print("    - /nbp/trajectory_path")
        print("    - /nbp/trajectory_poses")
        print("    - /nbp/trajectory_markers")
        print("    - /nbp/point_cloud_markers")

    def create_header(self):
        """创建ROS消息头"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        return header

    def publish_trajectory_path(self, trajectory):
        """
        发布轨迹为Path消息

        Args:
            trajectory: list of [x, y, z] positions
        """
        path = Path()
        path.header = self.create_header()

        for i, pos in enumerate(trajectory):
            pose_stamped = PoseStamped()
            pose_stamped.header = self.create_header()
            pose_stamped.pose.position.x = pos[0]
            pose_stamped.pose.position.y = pos[1]
            pose_stamped.pose.position.z = pos[2]
            pose_stamped.pose.orientation.w = 1.0
            path.poses.append(pose_stamped)

        self.path_pub.publish(path)
        print(f"Published trajectory path with {len(trajectory)} poses")

    def publish_trajectory_poses(self, pose_history):
        """
        发布轨迹姿态为PoseArray消息

        Args:
            pose_history: list of pose dicts with 'position' and 'quaternion'
        """
        pose_array = PoseArray()
        pose_array.header = self.create_header()

        for pose_dict in pose_history:
            pose = Pose()

            # 位置
            if 'position' in pose_dict:
                pos = pose_dict['position']
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = pos[2]

            # 姿态
            if 'quaternion' in pose_dict:
                quat = pose_dict['quaternion']
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
            else:
                pose.orientation.w = 1.0

            pose_array.poses.append(pose)

        self.poses_pub.publish(pose_array)
        print(f"Published {len(pose_history)} poses")

    def publish_trajectory_markers(self, trajectory, marker_size=0.1):
        """
        发布轨迹标记（球体）

        Args:
            trajectory: list of [x, y, z] positions
            marker_size: 标记大小
        """
        marker_array = MarkerArray()

        # 删除旧标记
        delete_marker = Marker()
        delete_marker.header = self.create_header()
        delete_marker.ns = "trajectory_points"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 创建球体标记
        for i, pos in enumerate(trajectory):
            marker = Marker()
            marker.header = self.create_header()
            marker.ns = "trajectory_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # 位置
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.w = 1.0

            # 大小
            marker.scale.x = marker_size
            marker.scale.y = marker_size
            marker.scale.z = marker_size

            # 颜色（渐变色：从绿到红）
            ratio = i / max(len(trajectory) - 1, 1)
            marker.color.r = ratio
            marker.color.g = 1.0 - ratio
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker.lifetime = rospy.Duration(0)  # 永久显示
            marker_array.markers.append(marker)

        # 创建连线
        line_marker = Marker()
        line_marker.header = self.create_header()
        line_marker.ns = "trajectory_line"
        line_marker.id = len(trajectory)
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        for pos in trajectory:
            point = Point()
            point.x = pos[0]
            point.y = pos[1]
            point.z = pos[2]
            line_marker.points.append(point)

        line_marker.scale.x = 0.02  # 线宽
        line_marker.color.r = 1.0
        line_marker.color.g = 0.5
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        line_marker.lifetime = rospy.Duration(0)

        marker_array.markers.append(line_marker)

        self.markers_pub.publish(marker_array)
        print(f"Published {len(trajectory)} trajectory markers")

    def publish_point_cloud_marker(self, ply_file, color=None, point_size=0.01):
        """
        从PLY文件发布点云标记

        Args:
            ply_file: PLY文件路径
            color: 点云颜色 [r, g, b, a]
            point_size: 点大小
        """
        if color is None:
            color = [0.5, 0.5, 0.5, 0.5]

        # 读取PLY文件（简单解析）
        points = []
        try:
            with open(ply_file, 'r') as f:
                lines = f.readlines()

                # 找到header结束位置
                header_end = 0
                num_vertices = 0
                for i, line in enumerate(lines):
                    if 'element vertex' in line:
                        num_vertices = int(line.split()[-1])
                    if 'end_header' in line:
                        header_end = i + 1
                        break

                # 读取点云数据
                for i in range(header_end, min(header_end + num_vertices, len(lines))):
                    parts = lines[i].strip().split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])

        except Exception as e:
            print(f"Error reading PLY file: {e}")
            return

        if len(points) == 0:
            print("No points loaded from PLY file")
            return

        # 创建点云标记
        marker = Marker()
        marker.header = self.create_header()
        marker.ns = "point_cloud"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        for point in points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)

        marker.scale.x = point_size
        marker.scale.y = point_size

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        marker.lifetime = rospy.Duration(0)

        self.point_cloud_pub.publish(marker)
        print(f"Published point cloud marker with {len(points)} points")

    def visualize_from_json(self, results_json_path, publish_rate=1.0):
        """
        从results.json文件读取并可视化轨迹

        Args:
            results_json_path: results.json文件路径
            publish_rate: 发布频率（Hz）
        """
        # 读取结果文件
        with open(results_json_path, 'r') as f:
            results = json.load(f)

        trajectory = results.get('trajectory', [])
        pose_history = results.get('pose_history', [])

        print(f"\nLoaded results from {results_json_path}")
        print(f"  Trajectory points: {len(trajectory)}")
        print(f"  Pose history: {len(pose_history)}")

        # 发布频率
        rate = rospy.Rate(publish_rate)

        print("\nPublishing to RViz...")
        print("Press Ctrl+C to stop\n")

        try:
            while not rospy.is_shutdown():
                # 发布轨迹
                if len(trajectory) > 0:
                    self.publish_trajectory_path(trajectory)
                    self.publish_trajectory_markers(trajectory)

                # 发布姿态
                if len(pose_history) > 0:
                    self.publish_trajectory_poses(pose_history)

                rate.sleep()

        except KeyboardInterrupt:
            print("\nVisualization stopped")


def visualize_results(results_dir, frame_id='map', publish_rate=1.0):
    """
    可视化实机测试结果

    Args:
        results_dir: 结果目录（包含results.json）
        frame_id: TF frame ID
        publish_rate: 发布频率
    """
    if not ROS_AVAILABLE:
        print("Error: ROS is not available")
        return

    # 查找results.json
    results_json = os.path.join(results_dir, 'results.json')
    if not os.path.exists(results_json):
        print(f"Error: results.json not found in {results_dir}")
        return

    # 创建可视化器
    visualizer = RVizTrajectoryVisualizer(frame_id=frame_id)

    # 等待发布器就绪
    rospy.sleep(1.0)

    # 可视化
    visualizer.visualize_from_json(results_json, publish_rate=publish_rate)

    # 可选：发布点云
    full_pc_ply = os.path.join(results_dir, 'full_pc.ply')
    if os.path.exists(full_pc_ply):
        print(f"\nPublishing point cloud from {full_pc_ply}")
        visualizer.publish_point_cloud_marker(
            full_pc_ply,
            color=[0.3, 0.7, 0.3, 0.3],
            point_size=0.02
        )


def create_rviz_config(output_file='nbp_visualization.rviz'):
    """
    创建RViz配置文件

    Args:
        output_file: 输出配置文件路径
    """
    config = """Panels:
  - Class: rviz/Displays
    Name: Displays
  - Class: rviz/Views
    Name: Views

Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz/Grid
      Enabled: true
      Name: Grid
      Plane: XY
      Plane Cell Count: 50
      Reference Frame: map

    - Alpha: 1
      Buffer Length: 1
      Class: rviz/Path
      Color: 255; 85; 0
      Enabled: true
      Line Style: Lines
      Line Width: 0.03
      Name: Trajectory Path
      Topic: /nbp/trajectory_path
      Value: true

    - Class: rviz/MarkerArray
      Enabled: true
      Name: Trajectory Markers
      Topic: /nbp/trajectory_markers
      Value: true

    - Class: rviz/Marker
      Enabled: true
      Name: Point Cloud
      Topic: /nbp/point_cloud_markers
      Value: true

  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
  Name: root
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 10
      Enable Stereo Rendering: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Name: Current View
      Pitch: 0.785
      Target Frame: map
      Yaw: 0.785
Window Geometry:
  Width: 1920
  Height: 1080
"""

    with open(output_file, 'w') as f:
        f.write(config)

    print(f"RViz config file created: {output_file}")
    print(f"To use it, run: rviz -d {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RViz Trajectory Visualization')
    parser.add_argument('results_dir', type=str, nargs='?', default=None,
                       help='Path to results directory (containing results.json)')
    parser.add_argument('--frame-id', type=str, default='map',
                       help='TF frame ID (default: map)')
    parser.add_argument('--rate', type=float, default=1.0,
                       help='Publishing rate in Hz (default: 1.0)')
    parser.add_argument('--create-config', action='store_true',
                       help='Create RViz config file and exit')

    args = parser.parse_args()

    # 创建配置文件
    if args.create_config:
        create_rviz_config()
        exit(0)

    # 可视化
    if args.results_dir is None:
        print("Error: results_dir is required")
        print("Usage: python rviz_visualizer.py <results_dir>")
        print("   or: python rviz_visualizer.py --create-config")
        exit(1)

    visualize_results(args.results_dir, args.frame_id, args.rate)
