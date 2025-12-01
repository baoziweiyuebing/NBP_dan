"""
Real Robot Testing Script for NBP Planning
实机测试脚本 - 基于RealSense D435i和Unitree Go2w
参考test_nbp_planning.py，实现实机环境下的自主探索
"""

import argparse
import os
import sys
import json
import time
import torch
import numpy as np
from datetime import datetime

# 添加路径
sys.path.append(os.path.abspath('./'))
from next_best_path.networks.nbp_model import NBP
from next_best_path.utility.utils import *
from real_world_env import RealWorldEnvironment, RobotPose
from Unigoal_Action_Mapping import init_Agent, get_state, get_current_pose, move_to_pose, move_along_path

dir_path = os.path.abspath(os.path.dirname(__file__))


def save_point_cloud_to_ply(points, colors=None, filename="point_cloud.ply"):
    """
    保存点云到PLY文件

    Args:
        points: Nx3 numpy数组或torch张量，点云坐标
        colors: Nx3 numpy数组或torch张量，点云颜色（可选）
        filename: 输出文件名
    """
    # 转换为numpy数组
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if colors is not None and torch.is_tensor(colors):
        colors = colors.cpu().numpy()

    # 检查点云是否为空
    if len(points) == 0:
        print(f"Warning: Cannot save empty point cloud to {filename}")
        return

    # 确保颜色在0-255范围内
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # 写入PLY文件
    with open(filename, 'w') as f:
        # 写入头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        # 写入数据
        for i in range(len(points)):
            if colors is not None:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} "
                       f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
            else:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")

    print(f"Point cloud saved to {filename}")


def test_real_robot_nbp_planning(
        nbp_weights_path,
        nbp_config_path,
        output_dir="./results/real_robot",
        n_poses=20,  # 实机测试用更少的poses
        camera_width=640,
        camera_height=480,
        device='cuda:0',
        robot_interface="enx607d099f16d2",
        save_interval=5,
        max_linear_velocity=0.2,
        max_angular_velocity=0.3):
    """
    实机NBP规划测试

    Args:
        nbp_weights_path: NBP模型权重路径
        nbp_config_path: NBP配置文件路径
        output_dir: 结果输出目录
        n_poses: 探索步数（实机建议20-50步）
        camera_width: 相机图像宽度
        camera_height: 相机图像高度
        device: PyTorch设备
        robot_interface: 机器狗网络接口名
        save_interval: 每隔多少步保存一次点云
        max_linear_velocity: 最大线速度(m/s)
        max_angular_velocity: 最大角速度(rad/s)
    """

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Results will be saved to: {run_output_dir}")

    # ========================================
    # 1. 初始化实机环境
    # ========================================
    print("\n=== Initializing Real World Environment ===")
    try:
        env = RealWorldEnvironment(
            camera_width=camera_width,
            camera_height=camera_height,
            camera_fps=30,
            device=device
        )
    except RuntimeError as e:
        print(f"Failed to initialize environment: {e}")
        return
    except Exception as e:
        print(f"Unexpected error during initialization: {e}")
        return

    # ========================================
    # 2. 初始化机器狗
    # ========================================
    print("\n=== Initializing Unitree Go2w Robot ===")
    try:
        sport_client, sport_mode_state_sub = init_Agent(ifname=robot_interface)
        print("Robot initialized successfully!")
    except Exception as e:
        print(f"Error: Failed to initialize robot: {e}")
        print("Please check:")
        print(f"  1. Robot is powered on")
        print(f"  2. Network interface '{robot_interface}' is correct")
        print(f"  3. Robot is connected to the network")
        env.stop()
        return

    # 等待获取初始状态
    print("Waiting for robot state...")
    time.sleep(1.0)
    state = get_state()
    if state is not None:
        env.update_robot_pose_from_state(state)
        print(f"Initial robot pose: {env.robot_pose.to_dict()}")
    else:
        print("Warning: Could not get initial robot state")

    # 重置世界坐标系原点
    env.reset_world_origin()

    # ========================================
    # 3. 加载NBP模型
    # ========================================
    print("\n=== Loading NBP Model ===")

    # 加载配置
    try:
        with open(nbp_config_path, 'r') as f:
            params_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {nbp_config_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        return

    # # 创建NBP模型
    # nbp = NBP(
    #     input_channels=5,  # 4 height slices + 1 trajectory
    #     output_channels_value=8,
    #     output_channels_obstacle=1
    # ).to(device)
    nbp = NBP(
        img_ch=5,  # 对应原来的input_channels=5
        output_ch1=8,  # 对应原来的output_channels_value=8
        output_ch2=1  # 对应原来的output_channels_obstacle=1
    ).to(device)

    # 加载权重
    try:
        if not os.path.exists(nbp_weights_path):
            print(f"Error: NBP weights file not found: {nbp_weights_path}")
            print("Please download the weights from the instructions in README.md")
            return

        checkpoint = torch.load(nbp_weights_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            nbp.load_state_dict(checkpoint['model_state_dict'])
        else:
            nbp.load_state_dict(checkpoint)

        nbp.eval()
        print(f"NBP model loaded from {nbp_weights_path}")
    except Exception as e:
        print(f"Error: Failed to load NBP model: {e}")
        print("Please check:")
        print("  1. Weights file is valid")
        print("  2. Model architecture matches the weights")
        print("  3. PyTorch version is compatible")
        return

    # ========================================
    # 4. 初始化探索变量
    # ========================================
    print("\n=== Starting Exploration ===")

    device_torch = torch.device(device)
    full_pc = torch.zeros(0, 3, device=device_torch)
    full_pc_colors = torch.zeros(0, 3, device=device_torch)

    trajectory_history = []  # 轨迹历史（世界坐标系）
    pose_history = []  # 位姿历史

    # NBP参数
    n_pieces = 4  # 点云切片数量
    pc2img_size = (256, 256)
    prediction_range = (-40, 40)  # 单位：米（相机坐标系的X-Z平面范围）
    value_map_size = (64, 64)

    # Y轴分bin范围（用于点云切片，世界坐标系）
    # 注意：这是初始范围，实际使用时会根据点云动态调整
    y_range_default = 4.0  # 默认Y轴范围（米）

    try:
        # ========================================
        # 5. 探索循环
        # ========================================
        for pose_i in range(n_poses + 1):
            print(f"\n{'='*60}")
            print(f"Step {pose_i}/{n_poses}")
            print(f"{'='*60}")

            # 更新机器人状态
            state = get_state()
            if state is not None:
                env.update_robot_pose_from_state(state)
                current_pose = get_current_pose(state)
                print(f"Current robot pose: pos={current_pose['position']}, yaw={current_pose['yaw']:.3f}")
            else:
                print("Warning: Could not get robot state")
                current_pose = {'position': np.array([0.0, 0.0, 0.0]), 'yaw': 0.0}

            # 保存当前位姿
            env.save_current_pose()
            pose_history.append(env.robot_pose.to_dict())

            # 捕获图像和深度
            print("Capturing image and depth...")
            images, depth, R, T = env.capture_image_with_pose(apply_filters=True)
            print(f"  Image shape: {images.shape}, Depth shape: {depth.shape}")

            # 计算局部点云（世界坐标系）
            print("Computing partial point cloud...")
            part_pc, part_pc_colors = env.compute_partial_point_cloud_world(
                depth, images, R, T, gathering_factor=0.05
            )
            print(f"  Partial PC: {len(part_pc)} points")

            # 累积点云
            if len(part_pc) > 0:
                full_pc = torch.vstack((full_pc, part_pc))
                if part_pc_colors is not None and len(part_pc_colors) > 0:
                    full_pc_colors = torch.vstack((full_pc_colors, part_pc_colors))
            else:
                print("  Warning: No valid points in this frame")

            print(f"  Total PC: {len(full_pc)} points")

            # 保存当前相机位置到轨迹
            camera_position = T.squeeze(0).cpu().numpy()
            trajectory_history.append(camera_position.tolist())

            # 定期保存点云
            if pose_i % save_interval == 0:
                partial_ply_path = os.path.join(run_output_dir, f"partial_{pose_i:03d}.ply")
                if len(part_pc) > 0:
                    save_point_cloud_to_ply(
                        part_pc.cpu().numpy(),
                        part_pc_colors.cpu().numpy() if part_pc_colors is not None else None,
                        partial_ply_path
                    )

            # 如果是最后一步，不需要规划下一步
            if pose_i >= n_poses:
                break

            # ========================================
            # 6. NBP规划
            # ========================================
            print("\nPlanning next move...")

            # 准备NBP输入：点云切片
            if len(full_pc) > 0:
                # 动态计算Y轴范围（基于当前点云）
                y_min_pc = full_pc[:, 1].min().item()
                y_max_pc = full_pc[:, 1].max().item()
                y_center = (y_min_pc + y_max_pc) / 2.0
                y_half_range = max((y_max_pc - y_min_pc) / 2.0, y_range_default / 2.0)

                # 创建bins（对称于中心）
                y_bins = torch.linspace(
                    y_center - y_half_range,
                    y_center + y_half_range,
                    n_pieces + 1,
                    device=device_torch
                )

                # 将点云按Y轴分成n_pieces
                bins = torch.bucketize(full_pc[:, 1], y_bins[:-1]) - 1
                # 处理边界情况（bins可能为-1或n_pieces）
                bins = torch.clamp(bins, 0, n_pieces - 1)
                full_pc_groups = [full_pc[bins == i] for i in range(n_pieces)]

                # 转换当前相机位姿用于投影
                # transform_points_to_n_pieces需要 [x, y, z, elevation, azimuth]
                camera_position = T.squeeze(0).to(device_torch)  # (3,) - 确保在正确设备上
                # 从机器狗的yaw计算azimuth（简化：假设elevation=0，azimuth=yaw转换为度）
                azimuth_deg = current_pose['yaw'] * 180.0 / np.pi
                elevation_deg = 0.0  # 假设相机水平

                camera_current_pose = torch.cat([
                    camera_position,
                    torch.tensor([elevation_deg, azimuth_deg], dtype=torch.float32, device=device_torch)
                ])  # (5,) = [x, y, z, elevation, azimuth]

                full_pc_images = []
                for i in range(n_pieces):
                    if len(full_pc_groups[i]) > 0:
                        try:
                            points_2d = transform_points_to_n_pieces(
                                full_pc_groups[i], camera_current_pose, device_torch, no_rotation=False
                            )
                            pc_img = map_points_to_n_imgs(
                                points_2d, pc2img_size, prediction_range, device_torch
                            )
                        except Exception as e:
                            print(f"Warning: Failed to process point cloud group {i}: {e}")
                            pc_img = torch.zeros(1, pc2img_size[0], pc2img_size[1], device=device_torch)
                    else:
                        pc_img = torch.zeros(1, pc2img_size[0], pc2img_size[1], device=device_torch)
                    full_pc_images.append(pc_img)

                full_pc_images = torch.cat(full_pc_images, dim=0)  # (4, H, W)
                current_pc_imgs = full_pc_images.unsqueeze(0)  # (1, 4, H, W)

                # 轨迹历史投影
                if len(trajectory_history) > 0:
                    try:
                        trajectory_tensor = torch.tensor(
                            trajectory_history, dtype=torch.float32, device=device_torch
                        )
                        trajectory_2d = transform_points_to_n_pieces(
                            trajectory_tensor, camera_current_pose, device_torch, no_rotation=False
                        )
                        trajectory_img = map_points_to_n_imgs(
                            trajectory_2d, pc2img_size, prediction_range, device_torch
                        )
                    except Exception as e:
                        print(f"Warning: Failed to process trajectory: {e}")
                        trajectory_img = torch.zeros(1, pc2img_size[0], pc2img_size[1], device=device_torch)
                else:
                    trajectory_img = torch.zeros(1, pc2img_size[0], pc2img_size[1], device=device_torch)

                current_trajectory_img = trajectory_img.unsqueeze(0)  # (1, 1, H, W)

                # 合并输入
                nbp_input = torch.cat((current_pc_imgs, current_trajectory_img), dim=1)  # (1, 5, H, W)

                # NBP预测
                with torch.no_grad():
                    predicted_value_map, predicted_obstacle_map = nbp(nbp_input)

                # 验证输出维度
                if predicted_value_map.shape != torch.Size([1, 8, 64, 64]):
                    print(f"Warning: Unexpected value_map shape: {predicted_value_map.shape}, expected [1, 8, 64, 64]")
                if predicted_obstacle_map.shape != torch.Size([1, 1, 256, 256]):
                    print(f"Warning: Unexpected obstacle_map shape: {predicted_obstacle_map.shape}, expected [1, 1, 256, 256]")

                # 简化的目标选择：选择价值最高的方向
                max_gain_map, _ = torch.max(predicted_value_map, dim=1, keepdim=True)  # (1, 1, 64, 64)

                # 找到最大价值位置
                max_value = max_gain_map.max()
                max_idx = max_gain_map.view(-1).argmax()
                max_row = (max_idx // value_map_size[1]).item()
                max_col = (max_idx % value_map_size[1]).item()

                print(f"  Max value: {max_value:.3f} at grid ({max_row}, {max_col})")

                # 将grid坐标转换为相机坐标系偏移
                # prediction_range = (-40, 40)，value_map_size = (64, 64)
                grid_center = value_map_size[0] // 2
                meters_per_pixel = (prediction_range[1] - prediction_range[0]) / value_map_size[0]

                # 相机坐标系下的偏移（图像坐标转实际坐标）
                offset_x_camera = (max_col - grid_center) * meters_per_pixel  # 左右
                offset_z_camera = (grid_center - max_row) * meters_per_pixel  # 前后（注意Y轴翻转）

                # 转换到世界坐标系（考虑机器人朝向）
                cos_yaw = np.cos(current_pose['yaw'])
                sin_yaw = np.sin(current_pose['yaw'])

                # 世界坐标系偏移
                offset_x_world = cos_yaw * offset_z_camera - sin_yaw * offset_x_camera
                offset_y_world = sin_yaw * offset_z_camera + cos_yaw * offset_x_camera

                # 限制移动距离（安全考虑）
                max_move_distance = 2.0  # 最大移动2米
                move_distance = np.sqrt(offset_x_world**2 + offset_y_world**2)
                if move_distance > max_move_distance:
                    scale = max_move_distance / move_distance
                    offset_x_world *= scale
                    offset_y_world *= scale
                    print(f"  Limiting move distance from {move_distance:.2f}m to {max_move_distance}m")

                # 计算目标位姿
                target_position = current_pose['position'].copy()
                target_position[0] += offset_x_world
                target_position[1] += offset_y_world

                target_yaw = current_pose['yaw']  # 保持相同朝向

                target_pose = {
                    'position': target_position,
                    'yaw': target_yaw
                }

                print(f"  Target pose: pos={target_position}, yaw={target_yaw:.3f}")
                print(f"  Move vector: ({offset_x_world:.2f}, {offset_y_world:.2f}) m")

                # ========================================
                # 7. 控制机器狗移动
                # ========================================
                print("\nMoving robot to target pose...")
                success, error = move_to_pose(
                    current_pose, target_pose, sport_client,
                    position_tolerance=0.15,
                    yaw_tolerance=0.15,
                    max_linear_velocity=max_linear_velocity,
                    max_angular_velocity=max_angular_velocity,
                    timeout=20.0,
                    verbose=True
                )

                if not success:
                    print(f"Warning: Failed to reach target pose, error={error}")
                else:
                    print("Successfully reached target pose!")

            else:
                print("Warning: No point cloud data, skipping planning")

            # 短暂停留以稳定
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nExploration interrupted by user")

    except Exception as e:
        print(f"\n\nError during exploration: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ========================================
        # 8. 保存结果
        # ========================================
        print("\n=== Saving Results ===")

        # 停止机器人
        sport_client.Move(vx=0, vy=0, vyaw=0)
        print("Robot stopped")

        # 保存完整点云
        if len(full_pc) > 0:
            full_pc_path = os.path.join(run_output_dir, "full_pc.ply")
            save_point_cloud_to_ply(
                full_pc.cpu().numpy(),
                full_pc_colors.cpu().numpy() if len(full_pc_colors) > 0 else None,
                full_pc_path
            )

        # 保存轨迹数据
        results = {
            'n_poses': len(trajectory_history),
            'trajectory': trajectory_history,
            'pose_history': pose_history,
            'timestamp': timestamp,
            'parameters': {
                'camera_width': camera_width,
                'camera_height': camera_height,
                'n_poses': n_poses,
                'max_linear_velocity': max_linear_velocity,
                'max_angular_velocity': max_angular_velocity
            }
        }

        results_path = os.path.join(run_output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")

        # 保存轨迹为PLY（可视化用）
        if len(trajectory_history) > 0:
            trajectory_array = np.array(trajectory_history)
            trajectory_path = os.path.join(run_output_dir, "trajectory.ply")
            save_point_cloud_to_ply(trajectory_array, None, trajectory_path)
            print(f"Trajectory saved to {trajectory_path}")

        # 停止环境
        env.stop()

        print("\n=== Exploration Complete ===")
        print(f"Total poses: {len(trajectory_history)}")
        print(f"Total points: {len(full_pc)}")
        print(f"Output directory: {run_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real robot NBP planning test')
    parser.add_argument('--nbp_weights', type=str,
                       default='./weights/nbp/AiMDoom_simple_best_val.pth',
                       help='Path to NBP model weights')
    parser.add_argument('--nbp_config', type=str,
                       default='./configs/nbp/nbp_default_training_config.json',
                       help='Path to NBP config file')
    parser.add_argument('--output_dir', type=str,
                       default='./results/real_robot',
                       help='Output directory for results')
    parser.add_argument('--n_poses', type=int, default=5,
                       help='Number of exploration steps')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='PyTorch device')
    parser.add_argument('--robot_interface', type=str, default='enx607d099f16d2',
                       help='Robot network interface name')
    parser.add_argument('--max_velocity', type=float, default=0.2,
                       help='Maximum linear velocity (m/s)')
    parser.add_argument('--max_angular', type=float, default=0.3,
                       help='Maximum angular velocity (rad/s)')

    args = parser.parse_args()

    test_real_robot_nbp_planning(
        nbp_weights_path=args.nbp_weights,
        nbp_config_path=args.nbp_config,
        output_dir=args.output_dir,
        n_poses=args.n_poses,
        device=args.device,
        robot_interface=args.robot_interface,
        max_linear_velocity=args.max_velocity,
        max_angular_velocity=args.max_angular
    )
