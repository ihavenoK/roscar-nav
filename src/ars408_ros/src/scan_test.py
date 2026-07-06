#!/usr/bin/env python3
"""虚拟 LaserScan 行人识别效果测试

用法:
  1. 先 roslaunch ars408_ros radar_nav.launch
  2. 再 python3 scan_test.py
  3. 派人站到雷达前方 1~2m 处，观察控制台输出
"""
import rospy
import math
from sensor_msgs.msg import LaserScan


class ScanTest:
    def __init__(self):
        rospy.init_node("scan_test", anonymous=True)
        self._count = 0
        rospy.Subscriber("/pedestrian_scan", LaserScan, self.cb)
        rospy.loginfo("监听 /pedestrian_scan, 按 Ctrl-C 退出")

    def cb(self, msg):
        self._count += 1
        clusters = self._find_clusters(msg)

        if not clusters:
            if self._count % 20 == 0:
                rospy.loginfo("[%d] 无行人检测", self._count)
            return

        for c in clusters:
            rospy.loginfo(
                "[%d] 行人 %.2fm  角度 %+.1f°  |  "
                "扇形 %d beam  |  x=%.2f y=%.2f",
                self._count,
                c['dist'], math.degrees(c['angle']),
                c['width'],
                c['x'], c['y'],
            )

    def _find_clusters(self, msg):
        """从 ranges 中提取连续非 max 的区域作为行人"""
        clusters = []
        in_cluster = False
        start_i = 0

        for i, r in enumerate(msg.ranges):
            if r < msg.range_max - 0.01:
                if not in_cluster:
                    start_i = i
                    in_cluster = True
            else:
                if in_cluster:
                    clusters.append((start_i, i - 1, msg))
                    in_cluster = False
        if in_cluster:
            clusters.append((start_i, len(msg.ranges) - 1, msg))

        result = []
        for si, ei, msg in clusters:
            beams = ei - si + 1
            dists = [msg.ranges[i] for i in range(si, ei + 1)]
            avg_dist = sum(dists) / len(dists)
            center_i = (si + ei) // 2
            angle = msg.angle_min + center_i * msg.angle_increment

            result.append({
                'dist': avg_dist,
                'angle': angle,
                'width': beams,
                'x': avg_dist * math.cos(angle),
                'y': avg_dist * math.sin(angle),
            })
        return result


if __name__ == "__main__":
    ScanTest()
    rospy.spin()
