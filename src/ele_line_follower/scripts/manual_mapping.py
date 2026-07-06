#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘控制建图脚本 (manual_mapping.py)
====================================
键盘遥控小车 + gmapping 建图 + 一键保存地图。

控制键:
  W/↑  前进           S/↓  后退
  A/←  左转           D/→  右转
  X    停止(清零所有速度)
  R    加速档位        E    减速档位
  M    保存当前地图   (调用 map_saver, 保存到 ~/map/ 目录)
  C    切换激光雷达滤波 (限用有效距离, 减少建图噪点)
  Q    退出并停车

建图须知:
  1. 先运行: roslaunch roscar_slam gmapping.launch use_visual_follower:=false
     或用配套的 start_mapping.sh 一键启动
  2. 本脚本仅负责键盘发布 /cmd_vel, 建图由 gmapping 节点自动完成
  3. 建图完成后按 M 保存地图, 地图文件保存在 ~/map/ 目录
  4. 建议: 慢速行驶, 多停留让 gmapping 回环检测收敛

方向修正 (针对本小车):
  - linear  默认取反 (W 前进)
  - angular 默认不反 (A 左转)
  可通过 ROS 参数手动调整: ~reverse_linear / ~reverse_steer

运行:
  rosrun ele_line_follower manual_mapping.py
  或配合参数:
  rosrun ele_line_follower manual_mapping.py _linear_speed:=0.15 _angular_speed:=0.6 _map_dir:=/home/gdut/map
"""

import rospy
import sys
import os
import signal
import subprocess
import termios
import tty
from geometry_msgs.msg import Twist


def get_key(settings):
    """非阻塞读取单个按键 (支持方向键)"""
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    if ch == '\x1b':
        ch2 = sys.stdin.read(1)
        ch3 = sys.stdin.read(1)
        return ch + ch2 + ch3
    return ch


def save_map(map_dir, map_name):
    """调用 map_server 的 map_saver 保存当前地图"""
    os.makedirs(map_dir, exist_ok=True)
    save_path = os.path.join(map_dir, map_name)
    rospy.loginfo("正在保存地图到: %s ...", save_path)
    try:
        # 调用 map_saver 节点 (rosrun map_server map_saver)
        result = subprocess.run(
            ['rosrun', 'map_server', 'map_saver', '-f', save_path],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            rospy.loginfo("地图保存成功!\n  %s.yaml\n  %s.pgm", save_path, save_path)
            rospy.loginfo("  使用: roslaunch start_roscar navigation.launch map_file:=%s.yaml", save_path)
        else:
            rospy.logerr("地图保存失败 (exit=%d): %s", result.returncode, result.stderr.strip())
    except subprocess.TimeoutExpired:
        rospy.logerr("地图保存超时 (15秒), 请检查 gmapping 是否运行")
    except FileNotFoundError:
        rospy.logerr("未找到 map_saver, 请确认 map_server 包已安装")
    except Exception as e:
        rospy.logerr("保存地图异常: %s", str(e))


def check_gmapping_running():
    """检查 gmapping 节点是否在运行"""
    try:
        result = subprocess.run(
            ['rosnode', 'list'], capture_output=True, text=True, timeout=3
        )
        return 'slam_gmapping' in result.stdout
    except Exception:
        return False


def main():
    rospy.init_node('manual_mapping', anonymous=False)

    # ========== 参数 ==========
    linear_speed   = rospy.get_param('~linear_speed',  0.15)   # m/s (建图建议慢速)
    angular_speed  = rospy.get_param('~angular_speed', 0.6)    # rad/s
    reverse_linear = rospy.get_param('~reverse_linear', False)  # C++ 驱动已处理方向, 无需再反转
    reverse_steer  = rospy.get_param('~reverse_steer',  False)  # 转向不反
    map_dir        = rospy.get_param('~map_dir', os.path.expanduser('~/map'))
    map_name       = rospy.get_param('~map_name', 'my_map')

    lin_sign = -1.0 if reverse_linear else 1.0
    ang_sign = -1.0 if reverse_steer else 1.0

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    # 检查 gmapping 状态
    if check_gmapping_running():
        rospy.loginfo("已检测到 slam_gmapping 节点运行中, 建图就绪")
    else:
        rospy.logwarn("未检测到 slam_gmapping 节点! 请确认 gmapping 已启动")
        rospy.logwarn("提示: roslaunch roscar_slam gmapping.launch use_visual_follower:=false")

    settings = termios.tcgetattr(sys.stdin)
    rospy.loginfo("=" * 60)
    rospy.loginfo("    键盘建图遥控已启动  (速度状态机模式)")
    rospy.loginfo("    W/↑ 前进     S/↓ 后退")
    rospy.loginfo("    A/← 左转     D/→ 右转  (转向保留前进速度)")
    rospy.loginfo("    X   停止     R 加速     E 减速")
    rospy.loginfo("    M   保存地图 (目录: %s)", map_dir)
    rospy.loginfo("    C   激光滤波切换 (暂未实现, 占位)")
    rospy.loginfo("    Q   退出并停车")
    rospy.loginfo("  当前档位: lin=%.2f m/s  ang=%.2f rad/s", linear_speed, angular_speed)
    rospy.loginfo("=" * 60)

    # 速度状态
    twist = Twist()
    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            key = get_key(settings)

            if key in ('w', 'W', '\x1b[A'):          # 前进
                twist.linear.x = lin_sign * linear_speed
                rospy.loginfo("▶ 前进  lin=%.2f  ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('s', 'S', '\x1b[B'):        # 后退
                twist.linear.x = -lin_sign * linear_speed
                rospy.loginfo("◀ 后退  lin=%.2f  ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('a', 'A', '\x1b[D'):        # 左转 (保留线速度)
                twist.angular.z = ang_sign * angular_speed
                rospy.loginfo("↺ 左转  lin=%.2f  ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('d', 'D', '\x1b[C'):        # 右转 (保留线速度)
                twist.angular.z = -ang_sign * angular_speed
                rospy.loginfo("↻ 右转  lin=%.2f  ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('x', 'X'):                  # 停止
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                rospy.loginfo("■ 停止")
            elif key in ('r', 'R'):                  # 加速
                linear_speed = min(0.5, linear_speed + 0.05)
                if abs(twist.linear.x) > 0.001:
                    twist.linear.x = (1 if twist.linear.x * lin_sign > 0 else -1) * lin_sign * linear_speed
                rospy.loginfo("▲ 加速 -> lin=%.2f", linear_speed)
            elif key in ('e', 'E'):                  # 减速
                linear_speed = max(0.05, linear_speed - 0.05)
                if abs(twist.linear.x) > 0.001:
                    twist.linear.x = (1 if twist.linear.x * lin_sign > 0 else -1) * lin_sign * linear_speed
                rospy.loginfo("▼ 减速 -> lin=%.2f", linear_speed)
            elif key in ('m', 'M'):                  # 保存地图
                rospy.loginfo("--- 保存地图 ---")
                # 先停车, 等地图稳定再保存
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                cmd_pub.publish(twist)
                rospy.sleep(1.0)  # 等地图更新收敛
                save_map(map_dir, map_name)
            elif key in ('c', 'C'):                  # 激光滤波切换 (占位)
                rospy.loginfo("激光滤波切换功能 (待实现)")
            elif key in ('q', 'Q', '\x03'):          # 退出
                rospy.loginfo("即将退出, 是否保存地图? (按 M 保存, 按 Q 确认退出)")
                # 给用户 3 秒时间按 M
                start_t = rospy.Time.now()
                while (rospy.Time.now() - start_t).to_sec() < 3.0:
                    ch = get_key(settings)
                    if ch in ('m', 'M'):
                        save_map(map_dir, map_name)
                        break
                    elif ch in ('q', 'Q', '\x03'):
                        break
                break
            else:
                pass  # 其他键忽略, 保持当前速度

            cmd_pub.publish(twist)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        # 退出时停车
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        for _ in range(5):
            if not rospy.is_shutdown():
                cmd_pub.publish(twist)
                rospy.sleep(0.1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rospy.loginfo("遥控退出, 已停车")


if __name__ == '__main__':
    main()
