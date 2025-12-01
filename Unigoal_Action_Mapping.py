import sys
import time
import math
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # 新增：导入通道初始化工具
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from queue import Queue
import threading

# for go2w
# 全局状态变量 - 用于存储msg_handler处理后的状态
latest_state = Queue(maxsize=1)  # 缓存最新的msg状态

current_action = 0    # 默认停止
keep_running = False
last_action_time = time.time()
action_lock = threading.Lock()  # 新增锁


def init_Agent(ifname="enx607d099f16d2", topic="rt/lf/sportmodestate"):
    """初始化机器人连接和状态订阅，核心是设置msg_handler处理状态"""
    # 初始化网络通道
    ChannelFactoryInitialize(0, ifname)

    # 第二步：初始化运动客户端
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    # 核心：定义msg_handler处理接收到的msg并缓存
    def msg_handler(msg: SportModeState_):
        """msg处理函数，所有状态信息都从这里获取"""
        # 仅保留最新的状态信息
        if latest_state.full():
            latest_state.get_nowait()
        # 将完整的msg存入队列，供其他函数使用
        latest_state.put(msg)
        # 可在此处添加额外的msg处理逻辑
        # print(f"msg_handler处理新状态: {msg.timestamp}")

    # update for go2w
    sport_mode_state_sub = ChannelSubscriber(topic, SportModeState_)
    sport_mode_state_sub.Init(msg_handler, 10)  # 初始化订阅者，建立与主题的连接, 10为队列大小

    return sport_client, sport_mode_state_sub


def send_action_to_Agent(action, sport_client, velocity=0.1, angle=math.pi / 12):
    """
    # action(Navila): int, 1-前进，2-左转，3-右转， 0-停止
    # action(Unigoal): int, 1-前进，2-左转，3-右转， 0-停止
    # go1
    udp, cmd: 由init_Agent初始化得到
    state: sdk.HighState, 用于接收状态数据
    # go2w
    sport_client: init_Agent 初始化得到，发送动作指令
    velocity: float, 前进速度
    angle: float, 转向角度
    """
    """发送动作指令到机器人"""
    if action == 1:
        sport_client.Move(vx=velocity, vy=0, vyaw=0)
    elif action == 2:
        sport_client.Move(vx=velocity, vy=0, vyaw=angle)
    elif action == 3:
        sport_client.Move(vx=velocity, vy=0, vyaw=-angle)
    else:
        sport_client.Move(vx=0, vy=0, vyaw=0)


def set_action(action):
    """设置当前动作，更新时间戳"""
    global current_action, last_action_time
    # current_action = action
    # last_action_time = time.time()
    with action_lock:  # 加锁
        current_action = action
        last_action_time = time.time()


def get_state():
    """从msg_handler处理后的队列中获取最新状态"""
    if not latest_state.empty():
        return latest_state.get()  # 返回完整的msg对象
    return None


def keep_sending_action(sport_client, velocity=0.1, angle=math.pi/12):
    """持续发送动作的线程函数"""
    global keep_running, current_action, last_action_time
    keep_running = True
    while keep_running:
        with action_lock:
            time_since_last = time.time() - last_action_time
            action = current_action
            if time_since_last > 1.0:
                action = 0
                # set_action(action)  # 同步更新current_action (update)
                current_action = action  # 直接更新，避免锁嵌套
                last_action_time = time.time()
            # 发送动作
        send_action_to_Agent(action, sport_client, velocity, angle)
        # time.sleep(0.002)  # 500Hz
        # time.sleep(0.02) # 50Hz
        # time.sleep(0.1)  # 10Hz
        time.sleep(0.05) # 20Hz，根据机器人响应调整


def get_current_pose(state):
    """
    从机器狗状态获取当前位姿

    Args:
        state: SportModeState_ 对象

    Returns:
        current_pose: dict with keys 'position' (x, y, z) and 'yaw'
    """
    if state is None:
        return {'position': np.array([0.0, 0.0, 0.0]), 'yaw': 0.0}

    try:
        # 从状态提取位置信息
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

        # 从IMU获取yaw角度
        if hasattr(state, 'imu_state') and hasattr(state.imu_state, 'rpy'):
            rpy = state.imu_state.rpy
            if isinstance(rpy, (list, tuple)) and len(rpy) >= 3:
                yaw = float(rpy[2])  # yaw is the third element
            else:
                yaw = 0.0
        else:
            yaw = 0.0

        return {'position': position, 'yaw': yaw}

    except Exception as e:
        print(f"Warning: Failed to get current pose: {e}")
        print(f"  State type: {type(state)}")
        return {'position': np.array([0.0, 0.0, 0.0]), 'yaw': 0.0}


def move_to_pose(current_pose, target_pose, sport_client,
                 position_tolerance=0.1, yaw_tolerance=0.1,
                 max_linear_velocity=0.3, max_angular_velocity=0.5,
                 control_frequency=20.0, timeout=30.0, verbose=True):
    """
    将机器狗从当前位姿移动到目标位姿（基于速度控制）

    这个函数实现了一个简单的位姿到速度的控制器，将NBP输出的目标位姿
    转换为机器狗的速度指令(vx, vy, vyaw)

    Args:
        current_pose: dict with keys 'position' (x, y, z) and 'yaw'
        target_pose: dict with keys 'position' (x, y, z) and 'yaw'
        sport_client: SportClient 对象，用于发送速度指令
        position_tolerance: 位置容差（米），默认0.1m
        yaw_tolerance: 角度容差（弧度），默认0.1 rad
        max_linear_velocity: 最大线速度（m/s），默认0.3m/s
        max_angular_velocity: 最大角速度（rad/s），默认0.5rad/s
        control_frequency: 控制频率（Hz），默认20Hz
        timeout: 超时时间（秒），默认30秒
        verbose: 是否打印详细信息

    Returns:
        success: bool，是否成功到达目标位姿
        final_error: dict，最终的位置和角度误差
    """

    if verbose:
        print(f"\n=== Moving to target pose ===")
        print(f"Current: pos={current_pose['position']}, yaw={current_pose['yaw']:.3f}")
        print(f"Target:  pos={target_pose['position']}, yaw={target_pose['yaw']:.3f}")

    # PID控制器参数
    kp_linear = 1.0      # 线速度比例增益
    kp_angular = 2.0     # 角速度比例增益

    # 控制循环
    dt = 1.0 / control_frequency
    start_time = time.time()
    iteration = 0

    # 初始化误差变量（避免超时时未定义）
    position_error = float('inf')
    yaw_error = float('inf')

    while True:
        # 检查超时
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            if verbose:
                print(f"Timeout! Failed to reach target in {timeout}s")
            sport_client.Move(vx=0, vy=0, vyaw=0)  # 停止
            return False, {'position_error': position_error, 'yaw_error': abs(yaw_error)}

        # 计算位置误差（世界坐标系）
        current_pos = current_pose['position'][:2]  # 只考虑x, y
        target_pos = target_pose['position'][:2]
        position_error_world = target_pos - current_pos
        position_error = np.linalg.norm(position_error_world)

        # 计算角度误差
        current_yaw = current_pose['yaw']
        target_yaw = target_pose['yaw']
        yaw_error = target_yaw - current_yaw

        # 归一化角度误差到 [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # 检查是否到达目标
        if position_error < position_tolerance and abs(yaw_error) < yaw_tolerance:
            if verbose:
                print(f"Target reached! pos_error={position_error:.3f}m, yaw_error={abs(yaw_error):.3f}rad")
            sport_client.Move(vx=0, vy=0, vyaw=0)  # 停止
            return True, {'position_error': position_error, 'yaw_error': abs(yaw_error)}

        # 策略1: 先转向，再前进（更安全的策略）
        if abs(yaw_error) > yaw_tolerance * 2:
            # 需要先调整朝向
            vx = 0.0
            vy = 0.0
            vyaw = np.clip(kp_angular * yaw_error, -max_angular_velocity, max_angular_velocity)

        else:
            # 朝向基本正确，可以前进
            # 将位置误差转换到机器人坐标系
            cos_yaw = math.cos(current_yaw)
            sin_yaw = math.sin(current_yaw)

            # 旋转矩阵（世界坐标系 -> 机器人坐标系）
            error_x_robot = cos_yaw * position_error_world[0] + sin_yaw * position_error_world[1]
            error_y_robot = -sin_yaw * position_error_world[0] + cos_yaw * position_error_world[1]

            # 计算速度指令
            vx = np.clip(kp_linear * error_x_robot, -max_linear_velocity, max_linear_velocity)
            vy = np.clip(kp_linear * error_y_robot, -max_linear_velocity * 0.5, max_linear_velocity * 0.5)  # 侧向速度限制更小
            vyaw = np.clip(kp_angular * yaw_error, -max_angular_velocity, max_angular_velocity)

        # # 发送速度指令
        # sport_client.Move(vx=float(vx), vy=float(vy), vyaw=float(vyaw))
        # 发送速度指令
        sport_client.Move(vx=float(vx), vy=float(vy), vyaw=float(vyaw))

        # 打印进度
        if verbose and iteration % 10 == 0:
            print(f"  iter={iteration}, pos_err={position_error:.3f}m, yaw_err={yaw_error:.3f}rad, "
                  f"cmd=(vx={vx:.2f}, vy={vy:.2f}, vyaw={vyaw:.2f})")

        iteration += 1

        # 更新当前位姿（从最新状态获取）
        # 注意：这里需要实时更新current_pose，应该从get_state()获取最新状态
        state = get_state()
        if state is not None:
            current_pose = get_current_pose(state)

        # 控制循环延时
        time.sleep(dt)


def move_along_path(path_poses, sport_client, **kwargs):
    """
    沿着路径移动（路径是一系列位姿）

    Args:
        path_poses: list of poses, each pose is a dict with keys 'position' and 'yaw'
        sport_client: SportClient 对象
        **kwargs: 传递给move_to_pose的其他参数

    Returns:
        success: bool，是否成功完成整条路径
        failed_at: int，如果失败，失败在第几个waypoint（如果成功则为-1）
    """
    print(f"\n=== Moving along path with {len(path_poses)} waypoints ===")

    # 获取初始位姿
    state = get_state()
    current_pose = get_current_pose(state)

    for i, target_pose in enumerate(path_poses):
        print(f"\n--- Waypoint {i+1}/{len(path_poses)} ---")

        success, error = move_to_pose(current_pose, target_pose, sport_client, **kwargs)

        if not success:
            print(f"Failed to reach waypoint {i+1}")
            return False, i

        # 更新当前位姿
        state = get_state()
        current_pose = get_current_pose(state)

        # 短暂停留
        time.sleep(0.5)

    print(f"\n=== Path completed successfully! ===")
    return True, -1