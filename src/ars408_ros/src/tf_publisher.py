#!/usr/bin/env python3
"""持续发布 base_link → radar_link 的静态 TF（10Hz）。

StaticTransformBroadcaster 只发布一次，时间戳固定后 tf2 持续报 TF_OLD_DATA。
改用 TransformBroadcaster + 定时器，始终用 rospy.Time.now() 发布，消除告警。

注意: camera_link 由 robot_model_visualization.launch 统一发布
      (base_footprint → camera_link)，此处不再重复发布以避免 TF 树冲突。
"""
import rospy
import tf2_ros
import geometry_msgs.msg


def publish_static_tf():
    rospy.init_node('sensor_tf_publisher')
    broadcaster = tf2_ros.TransformBroadcaster()

    # base_link -> radar_link
    t1 = geometry_msgs.msg.TransformStamped()
    t1.header.frame_id = 'base_link'
    t1.child_frame_id = 'radar_link'
    t1.transform.translation.x = 0.12   # 雷达中心X偏移: 12cm
    t1.transform.translation.y = 0.0
    t1.transform.translation.z = 0.315  # 雷达中心Z偏移: 31.5cm
    t1.transform.rotation.w = 1.0

    rospy.loginfo("TF配置: base_link→radar_link (%.2f, %.2f, %.2f)",
                  t1.transform.translation.x,
                  t1.transform.translation.y,
                  t1.transform.translation.z)

    def _publish(_event):
        now = rospy.Time.now()
        t1.header.stamp = now
        broadcaster.sendTransform(t1)

    rospy.Timer(rospy.Duration(0.1), _publish)  # 10Hz 持续发布
    rospy.spin()


if __name__ == '__main__':
    publish_static_tf()
