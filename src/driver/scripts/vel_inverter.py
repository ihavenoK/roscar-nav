#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
速度方向反转节点 (vel_inverter.py)
============================
本车硬件方向: 负 linear.x = 前进 (与 ROS 标准相反)

manual_mapping.py 通过 reverse_linear 参数已修正，
但 move_base 不知道这个方向反转，导致 2D Nav Goal 前进变倒退。

本节点在导航时插入 move_base 和 car_driver 之间，
将 move_base 的 cmd_vel 反转后再发给底盘驱动。

用法 (在 launch 文件中):
  <node pkg="driver" type="vel_inverter.py" name="vel_inverter" output="screen">
    <remap from="cmd_vel_in"  to="cmd_vel_nav"/>
    <remap from="cmd_vel_out" to="cmd_vel"/>
  </node>
"""

import rospy
from geometry_msgs.msg import Twist


class VelInverter:
    def __init__(self):
        rospy.init_node("vel_inverter", anonymous=False)

        self.pub = rospy.Publisher("cmd_vel_out", Twist, queue_size=1)
        self.sub = rospy.Subscriber("cmd_vel_in", Twist, self.callback, queue_size=1)

        rospy.loginfo("速度反转节点已启动: cmd_vel_in -> cmd_vel_out (linear.x 取反)")

    def callback(self, msg):
        msg.linear.x = -msg.linear.x  # 反转前进方向
        self.pub.publish(msg)


if __name__ == "__main__":
    try:
        VelInverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
