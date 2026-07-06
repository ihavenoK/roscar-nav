#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉特征约束下的 gmapping 建图 —— 视觉循迹节点 (visual_line_follower.py)
----------------------------------------------------------------------
用摄像头识别地面黑线, PID + 自适应速度控制小车沿黑线匀速行驶,
为 gmapping 建图提供"视觉路标"约束, 弥补电磁循迹缺失。

核心约束机制:
  1. 弯道自适应减速: |误差|越大线速度越低, 保证不冲出黑线 ->
     轨迹紧贴直线 -> 地图边缘平直, 不出现波浪形扭曲。
  2. 丢线超时停车: 连续丢线超过 lost_timeout 秒则停车, 防止乱跑污染地图。
  3. 就绪门控: 收到首帧图像后才允许动车, 避免启动瞬间用默认值猛冲。

替代: ele_line_follower.py (电磁循迹, 读 /ele_sensor)

话题:
  订阅: /usb_cam/image_raw          (sensor_msgs/Image)
  发布: /cmd_vel                    (geometry_msgs/Twist)
  发布: /line_follower/image        (sensor_msgs/Image, 调试用)
  发布: /line_follower/status       (std_msgs/String, 状态字符串)

关机时(Ctrl+C)自动停车并把地图保存到 ~/map/roscar_map。
----------------------------------------------------------------------
调参提示:
  - 若小车朝错误方向转, 把 Kp 改成负值即可。
  - 黑线太细/太粗: 调 min_pixel / v_thresh。
  - 只看近处地面: 增大 roi_y_ratio(如 0.6 表示只取图像下 40%)。
  - 弯道仍冲线: 增大 k_curve 或减小 v_min_ratio。
"""
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
import subprocess
import os
import rospkg


class VisualLineFollower:
    def __init__(self):
        self.bridge = CvBridge()

        # ---- 发布器 ----
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_pub = rospy.Publisher('/line_follower/image', Image, queue_size=1)
        self.status_pub = rospy.Publisher('/line_follower/status', String, queue_size=1)

        # ---- 订阅器 (buff_size 设大, 避免 640x480 丢帧) ----
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_raw', Image, self.image_cb,
            queue_size=1, buff_size=2 ** 24)

        # ---- 速度 / PID 参数 (rosparam 可调) ----
        self.v_forward = rospy.get_param('~v_forward', 0.12)   # 正常前进线速度 m/s
        self.v_lost    = rospy.get_param('~v_lost', 0.05)      # 丢线时前进速度
        self.Kp        = rospy.get_param('~Kp', 1.2)           # 比例
        self.Ki        = rospy.get_param('~Ki', 0.0)           # 积分
        self.Kd        = rospy.get_param('~Kd', 0.3)           # 微分
        self.omega_max = rospy.get_param('~omega_max', 0.8)    # 角速度上限 rad/s

        # ---- 自适应速度参数 (视觉约束核心) ----
        # 弯道(|error|大)自动减速: v = v_forward * max(v_min_ratio, 1 - k_curve*|err|)
        self.v_min_ratio = rospy.get_param('~v_min_ratio', 0.4)  # 弯道最低速度比例
        self.k_curve     = rospy.get_param('~k_curve', 1.0)      # 弯道减速增益

        # ---- 丢线保护参数 ----
        self.lost_timeout = rospy.get_param('~lost_timeout', 1.5)  # 丢线超时(秒), 超时停车

        # ---- 方向反转参数 (免改代码, 调试用) ----
        # reverse_steering: 转向反向 (黑线在右却往左转时设 true)
        # reverse_linear:   前后反向 (该前进却倒退时设 true)
        self.reverse_steering = rospy.get_param('~reverse_steering', False)
        self.reverse_linear   = rospy.get_param('~reverse_linear', False)

        # ---- 黑线检测参数 ----
        self.roi_y_ratio = rospy.get_param('~roi_y_ratio', 0.5)  # ROI 从图像高度 50% 到底部
        self.v_thresh    = rospy.get_param('~v_thresh', 80)      # HSV-V 通道阈值, 小于此为黑
        self.min_pixel   = rospy.get_param('~min_pixel', 50)     # 最少黑像素数, 否则判丢线

        # ---- 速度平滑参数 (防抖动) ----
        # 帧率抖动时 v/omega 阶跃 -> 小车抖动 -> 轨迹波浪 -> 地图边缘不平
        # 用斜率限制 + 独立 30Hz 定时器下发, 解耦控制环与图像环
        self.cmd_linear_target  = 0.0   # 图像环算出的目标线速度
        self.cmd_angular_target = 0.0   # 图像环算出的目标角速度
        self.cmd_linear_cur     = 0.0   # 实际下发的当前线速度(平滑后)
        self.cmd_angular_cur    = 0.0   # 实际下发的当前角速度(平滑后)
        self.linear_ramp  = rospy.get_param('~linear_ramp', 0.5)   # 线速度变化率上限 m/s^2
        self.angular_ramp = rospy.get_param('~angular_ramp', 3.0)  # 角速度变化率上限 rad/s^2
        self.cmd_lost_stop = False      # 丢线超时标志, 控制环读取

        # ---- 状态 ----
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.has_line = False
        self.map_saved = False
        self.ready = False                  # 收到首帧图像后才允许动
        self.lost_duration = 0.0            # 连续丢线累计时间
        self.last_log_time = 0.0            # 周期日志计时

        # ---- 地图缓存 (Ctrl+C 存图的关键) ----
        # 持续缓存最新一帧 /map, 关机时直接写文件,
        # 不依赖 gmapping / map_saver 还活着 (它们会比循迹节点先死)
        self.latest_map = None
        self.map_lock = None  # 延迟初始化(需在 init_node 后)
        self.map_received = False

        rospy.on_shutdown(self.on_shutdown)

        # ---- 独立控制环: 30Hz 平滑下发速度 ----
        # 帧率抖动/掉帧时仍能稳定控车, 不依赖图像回调频率
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / 30.0), self.control_loop)

        # ---- 订阅 /map 持续缓存 (Ctrl+C 时 gmapping 会先死, 必须提前缓存) ----
        import threading
        self.map_lock = threading.Lock()
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_cb, queue_size=1)

        rospy.loginfo("VisualLineFollower 已启动, 等待图像 /usb_cam/image_raw ...")

    # ===================== 图像回调: 黑线检测 + PID =====================
    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn("图像转换失败: %s", e)
            return

        # ---- 时间间隔 dt (带防护) ----
        # 优先用图像时间戳, 异常时 fallback 到 rospy.now, 防微分项算错
        now_img = msg.header.stamp.to_sec()
        now_node = rospy.Time.now().to_sec()
        now = now_img if now_img > 0 else now_node
        if self.last_time is None:
            self.last_time = now
            self.ready = True
            rospy.loginfo("收到首帧图像, 视觉循迹就绪, 开始沿黑线行驶")
            return
        dt = now - self.last_time
        # dt 异常保护: 过小/过大/负值都用经验值, 防止 PID 微分项爆炸
        if dt <= 0 or dt > 0.5:
            dt = 0.033
        self.last_time = now

        # ---- 1. ROI: 只取图像下半部分(地面区域) ----
        h, w = frame.shape[:2]
        roi = frame[int(h * self.roi_y_ratio):, :]

        # ---- 2. 黑线检测: HSV 的 V 通道, 越暗越黑 ----
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        mask = cv2.inRange(v_channel, np.array([0], dtype=np.uint8),
                           np.array([self.v_thresh], dtype=np.uint8))

        # 形态学开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # ---- 3. 质心 ----
        m = cv2.moments(mask)
        black_pixels = int(m['m00'])
        cx = None
        if black_pixels >= self.min_pixel:
            cx = m['m10'] / m['m00']
            self.has_line = True
        else:
            self.has_line = False

        # ---- 4. 误差 (归一化到 [-1, 1], 线在右为正) ----
        center_x = w / 2.0
        if self.has_line:
            error = (cx - center_x) / center_x
        else:
            # 丢线: 沿用上次方向, 幅度减半缓慢找回
            error = self.last_error * 0.5 if abs(self.last_error) > 0.05 else 0.0

        # ---- 5. PID ----
        self.integral += error * dt
        self.integral = max(-1.0, min(1.0, self.integral))   # 抗饱和
        deriv = (error - self.last_error) / dt
        omega = self.Kp * error + self.Ki * self.integral + self.Kd * deriv
        omega = max(-self.omega_max, min(self.omega_max, omega))
        self.last_error = error

        # ---- 6. 丢线计时 ----
        if self.has_line:
            self.lost_duration = 0.0
        else:
            self.lost_duration += dt

        # ---- 7. 自适应速度: 弯道减速 (视觉约束核心) ----
        # |error| 越大说明弯越急, 线速度越小, 避免冲出黑线导致轨迹波浪
        v_scale = max(self.v_min_ratio, 1.0 - self.k_curve * abs(error))
        v_cmd = self.v_forward * v_scale

        # ---- 8. 方向反转参数 ----
        steer_sign = -1.0 if not self.reverse_steering else 1.0
        linear_sign = 1.0 if not self.reverse_linear else -1.0

        # ---- 9. 计算目标速度 (不再直接发布, 交给 control_loop 平滑下发) ----
        if not self.ready:
            self.cmd_linear_target  = 0.0
            self.cmd_angular_target = 0.0
        elif self.lost_duration > self.lost_timeout:
            # 丢线超时: 停车等人工把车摆回线上, 不污染地图
            self.cmd_linear_target  = 0.0
            self.cmd_angular_target = 0.0
            self.cmd_lost_stop = True
        elif self.has_line:
            self.cmd_linear_target  = linear_sign * v_cmd
            self.cmd_angular_target = steer_sign * omega
            self.cmd_lost_stop = False
        else:
            # 短暂丢线: 缓慢直行沿用上次方向找回
            self.cmd_linear_target  = linear_sign * self.v_lost
            self.cmd_angular_target = steer_sign * omega
            self.cmd_lost_stop = False

        # ---- 10. 调试图像 + 状态 ----
        self.publish_debug(mask, cx, center_x, self.has_line, error, omega, v_cmd)
        self.publish_status(v_cmd, omega)

    # ===================== 独立控制环: 30Hz 平滑下发速度 =====================
    def control_loop(self, event):
        """独立于图像频率, 以 30Hz 固定频率平滑下发速度。
        用斜率限制防止 v/omega 阶跃导致小车抖动(轨迹波浪 -> 地图边缘不平)。
        丢线超时强制停车。"""
        dt = 0.033  # 30Hz
        if not self.ready or self.cmd_lost_stop:
            # 就绪前或丢线超时: 直接停车, 不走斜率(快速停止)
            target_v, target_w = 0.0, 0.0
        else:
            target_v, target_w = self.cmd_linear_target, self.cmd_angular_target

        # 斜率限制: 限制每个控制周期速度变化幅度
        max_dv = self.linear_ramp * dt
        max_dw = self.angular_ramp * dt
        dv = target_v - self.cmd_linear_cur
        dw = target_w - self.cmd_angular_cur
        dv = max(-max_dv, min(max_dv, dv))
        dw = max(-max_dw, min(max_dw, dw))
        self.cmd_linear_cur  += dv
        self.cmd_angular_cur += dw

        twist = Twist()
        twist.linear.x  = self.cmd_linear_cur
        twist.angular.z = self.cmd_angular_cur
        self.cmd_pub.publish(twist)

    # ===================== 调试图像 =====================
    def publish_debug(self, mask, cx, center_x, has_line, error, omega, v_cmd):
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (int(center_x), 0), (int(center_x), vis.shape[0]),
                 (0, 255, 255), 1)
        if has_line and cx is not None:
            cy = vis.shape[0] // 2
            cv2.circle(vis, (int(cx), cy), 6, (0, 0, 255), -1)
            cv2.line(vis, (int(center_x), cy), (int(cx), cy), (0, 255, 0), 2)
        cv2.putText(vis, "err=%.2f v=%.2f w=%.2f %s" % (error, v_cmd, omega,
                    "LINE" if has_line else "LOST"), (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
        except Exception:
            pass

    # ===================== 状态发布 (周期日志) =====================
    def publish_status(self, v_cmd, omega):
        now = rospy.Time.now().to_sec()
        if self.lost_duration > self.lost_timeout:
            state = "STOP(lost %.1fs)" % self.lost_duration
        elif self.has_line:
            state = "LINE"
        else:
            state = "LOST(%.1fs)" % self.lost_duration
        msg = String()
        msg.data = "state=%s v=%.2f w=%.2f err=%.2f" % (state, v_cmd, omega, self.last_error)
        self.status_pub.publish(msg)
        if now - self.last_log_time > 2.0:
            rospy.loginfo("[循迹] %s", msg.data)
            self.last_log_time = now

    # ===================== 地图缓存回调 =====================
    def map_cb(self, msg):
        """持续缓存最新一帧 /map。Ctrl+C 时 gmapping 会先于本节点退出,
        无法再请求地图, 所以必须提前缓存, 关机时直接写文件。"""
        with self.map_lock:
            self.latest_map = msg
            if not self.map_received:
                self.map_received = True
                rospy.loginfo("已收到首帧 /map (%dx%d), 地图缓存就绪",
                              msg.info.width, msg.info.height)

    # ===================== 关闭处理: 停车 + 保存地图 =====================
    def on_shutdown(self):
        if self.map_saved:
            return
        rospy.logwarn("检测到关闭, 停车并保存地图...")
        self.stop_robot()

        # ===== 修复: 直接从缓存写文件, 不依赖 gmapping/map_saver 还活着 =====
        # 原方案调 map_saver 请求 /map, 但 Ctrl+C 时 gmapping 先死, /map 没了 -> 超时失败
        self.save_map()
        self.map_saved = True

    def stop_robot(self):
        # 清零目标速度, 让控制环平滑减速到 0, 并强制多次下发零速度确保停车
        self.cmd_linear_target  = 0.0
        self.cmd_angular_target = 0.0
        self.cmd_lost_stop = True
        twist = Twist()
        for _ in range(10):
            if self.cmd_pub:
                self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
        rospy.loginfo("已停车")

    def save_map(self):
        """从内存缓存直接写 pgm + yaml, 完全不依赖任何其它节点存活。"""
        with self.map_lock:
            map_msg = self.latest_map

        if map_msg is None:
            rospy.logerr("未缓存到任何 /map, 无法保存。请确认建图期间 gmapping 正常运行过。")
            return

        map_dir = rospkg.RosPack().get_path('start_roscar') + "/map"
        map_path = os.path.join(map_dir, "roscar_map")
        os.makedirs(map_dir, exist_ok=True)
        pgm_path = map_path + ".pgm"
        yaml_path = map_path + ".yaml"
        for f in (pgm_path, yaml_path):
            if os.path.exists(f):
                os.remove(f)

        w = map_msg.info.width
        h = map_msg.info.height
        data = map_msg.data  # list[int], -1 未知, 0-100 占据概率
        res = map_msg.info.resolution
        ox = map_msg.info.origin.position.x
        oy = map_msg.info.origin.position.y
        oz = map_msg.info.origin.position.z

        # 占据阈值 (与 map_saver 默认一致)
        occ_thresh = 0.65
        free_thresh = 0.196

        # 转 pgm 像素: 未知205 / 占据0(黑) / 空闲254(白) / 中间127
        # pgm 第一行对应地图最高 y, 需上下翻转
        pixels = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                v = data[x + (h - y - 1) * w]
                if v < 0:
                    pixels[y, x] = 205
                elif v / 100.0 >= occ_thresh:
                    pixels[y, x] = 0
                elif v / 100.0 <= free_thresh:
                    pixels[y, x] = 254
                else:
                    pixels[y, x] = 127

        # 写 pgm (P5 二进制)
        with open(pgm_path, 'wb') as f:
            header = "P5\n%d %d\n255\n" % (w, h)
            f.write(header.encode('ascii'))
            f.write(pixels.tobytes())

        # 写 yaml (相对路径, 与 navigation.launch 加载一致)
        with open(yaml_path, 'w') as f:
            f.write("image: roscar_map.pgm\n")
            f.write("resolution: %f\n" % res)
            f.write("origin: [%f, %f, %f]\n" % (ox, oy, oz))
            f.write("negate: 0\n")
            f.write("occupied_thresh: %f\n" % occ_thresh)
            f.write("free_thresh: %f\n" % free_thresh)

        rospy.loginfo("地图保存完成: %s (%dx%d, res=%.3f)", pgm_path, w, h, res)


if __name__ == '__main__':
    try:
        rospy.init_node('visual_line_follower', anonymous=False)
        node = VisualLineFollower()
        rospy.spin()   # 由图像回调驱动, 无需额外循环
    except rospy.ROSInterruptException:
        pass
