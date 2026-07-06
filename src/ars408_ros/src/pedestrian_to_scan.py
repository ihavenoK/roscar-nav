#!/usr/bin/env python3
# coding=utf-8
"""
行人位置 → 虚拟 LaserScan 节点

订阅 fusion_person_detect 发布的行人位置（MarkerArray），
将每个确认的行人转换为对应的角度-距离扇形区域，
发布为 /pedestrian_scan 话题供 costmap 的 obstacle_layer 订阅。

优势 vs PointCloud2:
  - 更轻量（360 float vs 数百个 points）
  - 与 LiDAR scan 同格式，costmap 处理路径一致
  - "雷达标记，激光清除" 分工清晰
"""

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray


class PedestrianToScan:
    """行人 MarkerArray → 虚拟 LaserScan"""

    def __init__(self):
        rospy.init_node("pedestrian_to_scan")

        # ---- LaserScan 参数 ----
        self.scan_frame   = rospy.get_param("~frame_id", "camera_link")
        self.range_max    = rospy.get_param("~range_max", 8.0)
        self.range_min    = rospy.get_param("~range_min", 0.1)
        self.angle_min    = rospy.get_param("~angle_min", -math.pi)
        self.angle_max    = rospy.get_param("~angle_max", math.pi)
        self.angle_inc    = rospy.get_param("~angle_increment", math.radians(0.5))  # 0.5°
        self.person_radius  = rospy.get_param("~person_radius", 0.3)   # 行人包围圆半径(m)
        self.publish_rate   = rospy.get_param("~publish_rate", 10.0)

        # 预计算 beam 数量
        self.num_beams = int(
            (self.angle_max - self.angle_min) / self.angle_inc) + 1
        self._angle_min = self.angle_min
        self._angle_inc = self.angle_inc

        rospy.loginfo("pedestrian_to_scan: %d beams %.2f° res, frame=%s "
                      "person_radius=%.2fm",
                      self.num_beams, math.degrees(self.angle_inc),
                      self.scan_frame, self.person_radius)

        # ---- 最新行人数据 ----
        self.latest_persons = []  # [(x, y, z)] in scan_frame

        # ---- 订阅 / 发布 ----
        rospy.Subscriber("/fusion/person_markers", MarkerArray,
                         self.marker_cb, queue_size=5)
        self.pub_scan = rospy.Publisher(
            "/pedestrian_scan", LaserScan, queue_size=5)

        # ---- 定时器（保证持续发布，即使无行人） ----
        rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish)

    # ======================== 回调 ========================

    def marker_cb(self, msg):
        """解析 SPHERE 标记（ns="person"）提取行人位置"""
        persons = []
        for m in msg.markers:
            if m.ns != "person" or m.type != m.SPHERE or m.action != m.ADD:
                continue
            persons.append((
                m.pose.position.x,
                m.pose.position.y,
                m.pose.position.z,
            ))
        self.latest_persons = persons

    # ======================== 发布 ========================

    def publish(self, event=None):
        """生成并发布 LaserScan"""
        msg = LaserScan()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.scan_frame
        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_inc
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / self.publish_rate
        msg.range_min = self.range_min
        msg.range_max = self.range_max

        # 全部初始化为最大量程（无行人 = 无遮挡）
        ranges = [self.range_max] * self.num_beams

        for px, py, pz in self.latest_persons:
            # 2D 距离和角度（只考虑 XY 平面）
            dist = math.hypot(px, py)
            angle = math.atan2(py, px)

            # 该行人占据的角宽度
            half_angle = math.atan2(self.person_radius, max(dist, 0.01))

            # 对应的 beam 索引
            idx_start = int((angle - half_angle - self._angle_min)
                            / self._angle_inc)
            idx_end   = int((angle + half_angle - self._angle_min)
                            / self._angle_inc)

            idx_start = max(0, min(self.num_beams - 1, idx_start))
            idx_end   = max(0, min(self.num_beams - 1, idx_end))

            if idx_start > idx_end:
                idx_start, idx_end = idx_end, idx_start

            # 填入该扇区的距离（多个行人重叠时取最近）
            for i in range(idx_start, idx_end + 1):
                if dist < ranges[i]:
                    ranges[i] = dist

        msg.ranges = ranges
        msg.intensities = []  # 虚拟扫描无强度
        self.pub_scan.publish(msg)


if __name__ == "__main__":
    try:
        node = PedestrianToScan()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
