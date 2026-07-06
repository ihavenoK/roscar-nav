#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动遥控节点 (manual_teleop.py) —— 用于方案B: 人工遥控 + gmapping 建图
----------------------------------------------------------------------
键盘控制小车, 无需安装 teleop_twist_keyboard 包。

已内置方向修正 (针对本小车下位机方向反的问题):
  - linear  方向取反 (W 前进 / S 后退)
  - angular 方向取反 (A 左转 / D 右转)

速度模式: 速度状态机 (按一次设定方向并持续发布, 松手不会停)
  按 A/D 转向时保留当前线速度(不强制归零), 避免原地打转失控

控制键:
  W/↑  前进      S/↓  后退
  A/←  左转(保留前进速度)   D/→  右转(保留前进速度)
  X    停止(清零所有速度)
  R    加速档位  E    减速档位
  Q    退出并停车

运行:
  rosrun ele_line_follower manual_teleop.py
  或
  python3 manual_teleop.py _linear_speed:=0.2 _angular_speed:=0.8
"""
import rospy
import sys
import termios
import tty
from geometry_msgs.msg import Twist


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    ch = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    # 处理方向键转义序列 ESC[A/B/C/D
    if ch == '\x1b':
        ch2 = sys.stdin.read(1)
        ch3 = sys.stdin.read(1)
        return ch + ch2 + ch3
    return ch


def main():
    rospy.init_node('manual_teleop', anonymous=False)

    # 参数
    linear_speed  = rospy.get_param('~linear_speed', 0.2)   # m/s
    angular_speed = rospy.get_param('~angular_speed', 0.8)  # rad/s
    # 方向修正 (本小车下位机: linear 反, angular 不反)
    reverse_linear = rospy.get_param('~reverse_linear', False)
    reverse_steer  = rospy.get_param('~reverse_steer', False)
    lin_sign = -1.0 if reverse_linear else 1.0
    ang_sign = -1.0 if reverse_steer else 1.0

    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    settings = termios.tcgetattr(sys.stdin)
    rospy.loginfo("=" * 55)
    rospy.loginfo("手动遥控已启动 (方向已内置修正, 速度状态机模式)")
    rospy.loginfo("  W/↑ 前进   S/↓ 后退")
    rospy.loginfo("  A/← 左转   D/→ 右转  (转向时保留前进速度)")
    rospy.loginfo("  X   停止")
    rospy.loginfo("  R   加速   E   减速")
    rospy.loginfo("  Q   退出并停车")
    rospy.loginfo("当前档位: lin=%.2f ang=%.2f", linear_speed, angular_speed)
    rospy.loginfo("=" * 55)

    # 速度状态 (持续发布, 松手不停)
    twist = Twist()
    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            key = get_key(settings)

            if key in ('w', 'W', '\x1b[A'):      # 前进
                twist.linear.x = lin_sign * linear_speed
                rospy.loginfo("前进  lin=%.2f ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('s', 'S', '\x1b[B'):    # 后退
                twist.linear.x = -lin_sign * linear_speed
                rospy.loginfo("后退  lin=%.2f ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('a', 'A', '\x1b[D'):    # 左转 (保留当前线速度)
                twist.angular.z = ang_sign * angular_speed
                rospy.loginfo("左转  lin=%.2f ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('d', 'D', '\x1b[C'):    # 右转 (保留当前线速度)
                twist.angular.z = -ang_sign * angular_speed
                rospy.loginfo("右转  lin=%.2f ang=%.2f", twist.linear.x, twist.angular.z)
            elif key in ('x', 'X'):
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                rospy.loginfo("停止")
            elif key in ('r', 'R'):
                linear_speed = min(0.5, linear_speed + 0.05)
                # 同步更新当前前进/后退速度
                if abs(twist.linear.x) > 0.001:
                    twist.linear.x = (1 if twist.linear.x*lin_sign > 0 else -1) * lin_sign * linear_speed
                rospy.loginfo("加速 -> lin=%.2f", linear_speed)
            elif key in ('e', 'E'):
                linear_speed = max(0.05, linear_speed - 0.05)
                if abs(twist.linear.x) > 0.001:
                    twist.linear.x = (1 if twist.linear.x*lin_sign > 0 else -1) * lin_sign * linear_speed
                rospy.loginfo("减速 -> lin=%.2f", linear_speed)
            elif key in ('q', 'Q', '\x03'):      # Q 或 Ctrl+C
                break
            else:
                pass  # 其他键忽略, 保持当前速度

            # 持续发布当前速度状态
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
