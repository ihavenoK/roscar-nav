#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
里程计方向反转节点 (odom_inverter.py)
============================
本车硬件方向: 负 linear.x = 前进 (与 ROS 标准相反)

car_driver 发布的 /odom 中 linear.x 方向与 ROS 标准相反，
导致 EKF/AMCL 认为车在倒退，RViz 模型显示错误。

此节点订阅 /odom_raw (driver 原始数据),
取反 linear.x 后发布到 /odom (ROS 标准方向),
确保所有下游节点 (EKF, AMCL, RViz) 获得正确的方向。

用法 (在 launch 文件中):
  <node pkg="driver" type="odom_inverter.py" name="odom_inverter" output="screen">
    <remap from="odom_in"  to="odom_raw"/>
    <remap from="odom_out" to="odom"/>
  </node>
"""

import rospy
from nav_msgs.msg import Odometry


class OdomInverter:
    def __init__(self):
        rospy.init_node("odom_inverter", anonymous=False)

        self.pub = rospy.Publisher("odom_out", Odometry, queue_size=10)
        self.sub = rospy.Subscriber("odom_in", Odometry, self.callback, queue_size=10)

        rospy.loginfo("里程计反转节点已启动: odom_in -> odom_out (linear.x 取反)")

    def callback(self, msg):
        msg.twist.twist.linear.x = -msg.twist.twist.linear.x
        # 只反转线速度 linear.x，角速度 angular.z 不反转
        # (与 manual_mapping.py 的 reverse_linear=True, reverse_steer=False 保持一致)
        self.pub.publish(msg)


if __name__ == "__main__":
    try:
        OdomInverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
