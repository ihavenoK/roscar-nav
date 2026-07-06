#!/usr/bin/env python3
# coding=utf-8
"""
毫米波雷达 + YOLO 数据融合节点

功能:
  1. 雷达点云 + YOLO检测 → TF投影 + bbox关联
  2. 2D Kalman 滤波器追踪行人(CV模型)
  3. 输出: 融合图片 /fusion/printpoint；行人位置 PointStamped；MarkerArray；散点图
"""
import rospy
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tf2_ros
from collections import deque
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2


class SimpleKalman2D:
    """2D卡尔曼滤波（radar_link XY平面），CV模型
    状态向量: [x, y, vx, vy]
    观测量:   [x, y]（位置）
    """

    def __init__(self, process_noise=0.5, measure_noise=1.0):
        self.x = np.zeros(4)
        self.P = np.diag([10.0, 10.0, 5.0, 5.0])
        self.q = process_noise
        self.r = measure_noise
        self.initialized = False

    def predict(self, dt):
        if not self.initialized:
            return
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * self.q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_x, z_y, z_vx=None, z_vy=None):
        """位置观测更新。z_vx/z_vy 保留接口兼容，当前始终用 2D 观测。"""
        z = np.array([z_x, z_y])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.diag([self.r, self.r])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.initialized = True

    def get_position(self):
        return (self.x[0], self.x[1]) if self.initialized else None

    def get_velocity(self):
        return (self.x[2], self.x[3]) if self.initialized else None

    def get_predicted_position(self, dt=0.1):
        return (self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt) \
            if self.initialized else None

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.diag([10.0, 10.0, 5.0, 5.0])
        self.initialized = False


class FusionNode:
    def __init__(self):
        rospy.init_node('fusion_node')

        # -- 参数 --
        self.fx = rospy.get_param('~fx', 400.0)
        self.fy = rospy.get_param('~fy', 400.0)
        self.cx = rospy.get_param('~cx', 320.0)
        self.cy = rospy.get_param('~cy', 240.0)
        self.max_radar_dist = rospy.get_param('~max_radar_dist', 50.0)
        self.association_threshold = rospy.get_param('~association_threshold', 150.0)
        self.max_jump_dist = rospy.get_param('~max_jump_dist', 1.5)

        # -- CV Bridge & TF --
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("等待TF变换 radar_link → camera_link ...")
        try:
            self.tf_buffer.lookup_transform('camera_link', 'radar_link',
                                            rospy.Time(0), rospy.Duration(5.0))
            rospy.loginfo("TF变换获取成功!")
        except Exception as e:
            rospy.logwarn(f"TF变换暂不可用: {e}")

        # -- 缓存队列 --
        self.radar_buffer = deque(maxlen=50)
        self.yolo_buffer = deque(maxlen=50)
        self.max_time_diff = rospy.Duration(rospy.get_param('~max_time_diff', 0.5))

        # 订阅
        self.radar_sub = rospy.Subscriber('/radar/pointcloud', PointCloud2,
                                          self.radar_callback, queue_size=30)
        self.yolo_sub = rospy.Subscriber('/yolo/person_detections', Image,
                                         self.yolo_callback, queue_size=30)
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image,
                                          self.image_callback, queue_size=5)
        self.latest_image = None

        self.process_timer = rospy.Timer(rospy.Duration(0.1), self.process_callback)
        self._debug_counter = 0

        # -- 发布器 --
        self.person_pub = rospy.Publisher('/fusion/person_position', PointStamped, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/fusion/trajectory', Marker, queue_size=10)
        self.current_pos_pub = rospy.Publisher('/fusion/current_position', Marker, queue_size=10)
        self.printpoint_pub = rospy.Publisher('/fusion/printpoint', Image, queue_size=5)
        self.printtrajectory_pub = rospy.Publisher('/fusion/printtrajectory', Image, queue_size=5)
        self.person_radar_pub = rospy.Publisher('/fusion/person_radar_markers', MarkerArray, queue_size=10)
        self.raw_scatter_pub = rospy.Publisher('/fusion/raw_scatter', MarkerArray, queue_size=10)
        self.scatter_canvas_pub = rospy.Publisher('/fusion/scatter_canvas', Image, queue_size=5)
        self.trajectory_canvas_pub = rospy.Publisher('/fusion/trajectory_canvas', Image, queue_size=5)

        # -- 轨迹历史 --
        self.trajectory_points = []
        self.max_trajectory_points = 600
        self.trajectory_id = 0
        self.trajectory_count = 0

        # -- 跟踪状态 --
        self.last_position_radar = None
        self.kf = SimpleKalman2D(process_noise=0.5, measure_noise=1.5)
        self.confidence = 0
        self.confidence_threshold = 3
        self.lost_count = 0
        self.max_lost = 10

        self._last_process_time = None
        self._last_sim_time = None

        # -- 新息自适应 --
        self._innov_window = deque(maxlen=10)

        # -- 原始散点记录（不经滤波） --
        self.raw_scatter_points = []   # [(Xr, Yr, Zr, u_p, v_p)]
        self.max_scatter_points = 20
        self._last_scatter_time = None  # 上次记录散点的sim_time

        # -- 图像标注缓存 --
        self._latest_detections = []
        self._latest_associated = None
        self._latest_distance = None
        self._cached_R = None
        self._cached_t = None

        rospy.loginfo("FusionNode ready (KF: CV model, Vx gated, adaptive Q/R)")

    # -- 回调 --

    def radar_callback(self, msg):
        self.radar_buffer.append((msg.header.stamp, msg))

    def yolo_callback(self, msg):
        stamp = msg.header.stamp
        self.yolo_buffer.append((stamp, msg))

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    # -- 主处理 --

    def process_callback(self, event):
        if not self.radar_buffer or not self.yolo_buffer:
            return
        self._debug_counter += 1

        current_sim_time = rospy.Time.now()
        if self._last_sim_time is not None:
            if (current_sim_time - self._last_sim_time).to_sec() < -1.0:
                rospy.logwarn("[回跳检测] sim_time 回跳 %.1fs，清空 buffer 和跟踪状态",
                             (current_sim_time - self._last_sim_time).to_sec())
                self.radar_buffer.clear()
                self.yolo_buffer.clear()
                self._clear_state(full_reset=True)
                self._last_process_time = None
                self._last_sim_time = current_sim_time
                return
        self._last_sim_time = current_sim_time

        now_time = rospy.Time.now()
        if self._last_process_time is not None:
            actual_dt = (now_time - self._last_process_time).to_sec()
            if 0.02 < actual_dt < 1.0:
                self._current_dt = actual_dt
        else:
            self._current_dt = 0.1
        self._last_process_time = now_time

        max_diff_sec = self.max_time_diff.to_sec()
        best_radar = best_yolo = best_diff = None

        r_list = list(self.radar_buffer)
        y_list = list(self.yolo_buffer)
        ri, yi = 0, 0
        while ri < len(r_list) and yi < len(y_list):
            rs, rm = r_list[ri]
            ys, ym = y_list[yi]
            diff = (ys - rs).to_sec()
            abs_diff = abs(diff)
            if abs_diff <= max_diff_sec:
                if best_diff is None or abs_diff < abs(best_diff):
                    best_diff = diff
                    best_radar = (rs, rm)
                    best_yolo = (ys, ym)
            if rs < ys:
                ri += 1
            else:
                yi += 1

        if best_radar is None:
            self.lost_count += 1
            self._handle_lost()
            self._publish_cleared(Header())
            return

        _, radar_msg = best_radar
        _, yolo_msg = best_yolo

        best_r_stamp = best_radar[0]
        best_y_stamp = best_yolo[0]
        while self.radar_buffer and self.radar_buffer[0][0] <= best_r_stamp:
            self.radar_buffer.popleft()
        while self.yolo_buffer and self.yolo_buffer[0][0] <= best_y_stamp:
            self.yolo_buffer.popleft()

        header = Header(stamp=radar_msg.header.stamp, frame_id='camera_link')
        dt = best_diff

        if self._debug_counter % 30 == 0:
            rospy.loginfo(f"[同步] radar={best_r_stamp.to_sec():.3f} "
                         f"yolo={best_y_stamp.to_sec():.3f} "
                         f"dt={dt*1000:.1f}ms "
                         f"frame_dt={self._current_dt*1000:.0f}ms")

        # 1. YOLO检测
        detections = self._parse_yolo(yolo_msg)
        if not detections:
            self.lost_count += 1
            self._handle_lost()
            self._clear_state()
            self._publish_cleared(header)
            self._draw_overlay(header)
            return

        # 2. 雷达点云
        radar_points = self._parse_radar(radar_msg)
        if not radar_points:
            self.lost_count += 1
            self._handle_lost()
            self._clear_state()
            self._publish_cleared(header)
            self._draw_overlay(header)
            return

        # 3. TF
        tf = self._get_tf(radar_msg.header.stamp)
        if tf is None:
            self.lost_count += 1
            self._handle_lost()
            self._clear_state()
            self._publish_cleared(header)
            self._draw_overlay(header)
            return
        R, t = tf
        self._cached_R = R
        self._cached_t = t

        # 4. 投影 + 运动补偿 + bbox过滤 + bbox距离一致性预过滤
        projected = []
        for Xr, Yr, Zr, Vx, Vy in radar_points:
            comp_Xr = Xr + Vx * dt
            comp_Yr = Yr + Vy * dt
            Xc = R[0,0]*comp_Xr + R[0,1]*comp_Yr + R[0,2]*Zr + t[0]
            Yc = R[1,0]*comp_Xr + R[1,1]*comp_Yr + R[1,2]*Zr + t[1]
            Zc = R[2,0]*comp_Xr + R[2,1]*comp_Yr + R[2,2]*Zr + t[2]
            if Xc <= 0.5:
                continue
            u = self.fx * (-Yc / Xc) + self.cx
            v = self.fy * (-Zc / Xc) + self.cy
            matched_det = None
            for det in detections:
                bbox_area = det['width'] * det['height']
                margin = 40 + max(0, min(80, (12000 - bbox_area) * 0.008))
                if abs(u - det['center_x']) < det['width']/2 + margin and \
                   abs(v - det['center_y']) < det['height']/2 + margin:
                    matched_det = det
                    break
            if matched_det is not None and 0 <= u < 640 and 0 <= v < 480:
                # bbox距离一致性：雷达点距离必须在bbox估算距离的±45%内
                cand_dist = np.sqrt(Xr**2 + Yr**2)
                bbox_dist_est = self.fy * 1.7 / matched_det['height']  # PERSON_HEIGHT=1.7m
                if abs(cand_dist - bbox_dist_est) > bbox_dist_est * 0.45:
                    continue
                projected.append((comp_Xr, comp_Yr, Zr, Xc, Yc, Zc, u, v, Vx, Vy))

        self._publish_person_radar(header, projected)

        # 5. 关联
        best = self._associate(detections, projected)
        if best is None:
            self.lost_count += 1
            self._handle_lost()
            self._latest_associated = None
            self._latest_distance = None
            self._draw_overlay(header)
            return

        Xr, Yr, Zr, Xc, Yc, Zc, u_p, v_p, Vx_raw, Vy_raw = best
        r_dist = np.sqrt(Xr**2 + Yr**2)  # 一次性计算，后续复用

        self._latest_detections = detections
        self._latest_distance = r_dist

        # 6. 跳变过滤
        if self.last_position_radar is not None and self.lost_count < self.max_lost:
            lx, ly, _ = self.last_position_radar
            jump = np.sqrt((Xr - lx)**2 + (Yr - ly)**2)
            effective_max_jump = self.max_jump_dist + max(0, (r_dist - 5.0) * 0.2)
            if jump > effective_max_jump:
                self.lost_count += 1
                self._handle_lost()
                self._draw_overlay(header)
                return

        self.last_position_radar = (Xr, Yr, Zr)
        self.lost_count = 0
        self.confidence = min(self.confidence + 1, self.confidence_threshold + 5)

        # ★ 原始散点记录（跳变过滤后的关联点，不经KF，每1.35秒采样一个，全程最多20个）
        if len(self.raw_scatter_points) < self.max_scatter_points:
            now_t = rospy.Time.now().to_sec()
            if self._last_scatter_time is None or \
               (now_t - self._last_scatter_time) >= 1.35:
                self.raw_scatter_points.append((Xr, Yr, Zr, int(u_p), int(v_p)))
                self._last_scatter_time = now_t

        # 7. KF + 观测前推 + 新息自适应 → 黄点
        dt_kf = self._current_dt
        comp_factor = max(0.04, min(0.65, (0.55 * r_dist - 0.85) * 0.5))

        # v25d: 速度自适应damp——低速时压低前推量，防止行人停下时速度噪声导致黄点乱跳
        speed = np.sqrt(Vx_raw**2 + Vy_raw**2)
        speed_damp = max(0.0, min(1.0, (speed - 0.12) / 0.33))
        comp_factor *= speed_damp

        # v25b: 观测前推——把测量值推到"未来"，让KF拿到已补偿的观测
        Xr_comp = Xr + Vx_raw * dt_kf * comp_factor
        Yr_comp = Yr + Vy_raw * dt_kf * comp_factor

        # v25c: 新息自适应Q——机动时临时放大Q，匀速时回落
        kf_pred_pos = self.kf.get_predicted_position(dt=dt_kf)
        if kf_pred_pos is not None and self.kf.initialized:
            innov = np.sqrt((Xr_comp - kf_pred_pos[0])**2 + (Yr_comp - kf_pred_pos[1])**2)
        else:
            innov = 0.0
        self._innov_window.append(innov)
        avg_innov = sum(self._innov_window) / len(self._innov_window) if self._innov_window else 0.0

        q_base = 0.01 + (r_dist / 8.0) ** 2
        q_boost = 1.0 + min(3.0, max(0, avg_innov - 0.3) * 5.0)
        self.kf.q = q_base * q_boost
        self.kf.r = max(0.05, 0.50 - r_dist * 0.08)

        self.kf.predict(dt_kf)
        # ★ v27: Vx门控速度观测——径向速度>0.3m/s才启用，横移时退化为纯位置KF
        use_vx = abs(Vx_raw) > 0.3
        self.kf.update(Xr_comp, Yr_comp, z_vx=Vx_raw if use_vx else None, z_vy=None)
        kf_p = self.kf.get_position()
        kf_x, kf_y = kf_p if kf_p else (Xr, Yr)

        # v28: 去掉输出前推，黄点直接用KF输出，不再二次速度补偿
        kf_x_comp = kf_x
        kf_y_comp = kf_y

        # Z方向：优先用雷达实测Zr，钳位到[0.05, 0.5]防止异常值
        Zr_use = min(Zr, 0.5) if Zr > 0.05 else 0.5
        KF_xc = R[0,0]*kf_x_comp + R[0,1]*kf_y_comp + R[0,2]*Zr_use + t[0]
        KF_yc = R[1,0]*kf_x_comp + R[1,1]*kf_y_comp + R[1,2]*Zr_use + t[1]
        KF_zc = R[2,0]*kf_x_comp + R[2,1]*kf_y_comp + R[2,2]*Zr_use + t[2]

        # 黄点 = KF平滑 + 速度前推
        kf_u = self.fx * (-KF_yc / KF_xc) + self.cx if KF_xc > 0.5 else u_p
        kf_v = self.fy * (-KF_zc / KF_xc) + self.cy if KF_xc > 0.5 else v_p
        self._latest_associated = (kf_u, kf_v, kf_x_comp, kf_y_comp, Zr_use, Vx_raw, Vy_raw)
        self._latest_distance = np.sqrt(kf_x_comp**2 + kf_y_comp**2)

        # 8. 发布轨迹
        if self.confidence >= self.confidence_threshold:
            self._publish_person(header, (KF_xc, KF_yc, KF_zc))
            self._update_trajectory(header, (kf_x_comp, kf_y_comp, Zr_use))

        # 9. 绘制图像叠加
        self._draw_overlay(header)
        self._draw_trajectory_overlay(header)

        # 10. 发布原始散点到RViz
        self._publish_raw_scatter(header)

        # 11. 绘制白底坐标轴画布
        self._draw_scatter_canvas(header)
        self._draw_trajectory_canvas(header)

    # -- 行人雷达点发布 --

    def _publish_person_radar(self, header, projected):
        ma = MarkerArray()
        for i, (Xr, Yr, Zr, _, _, _, _, _, _, _) in enumerate(projected):
            m = Marker()
            m.header = header
            m.header.frame_id = 'radar_link'
            m.ns = 'person_radar'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = Xr
            m.pose.position.y = Yr
            m.pose.position.z = Zr
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.4
            m.color.r = 1.0; m.color.g = 0.3; m.color.b = 0.0; m.color.a = 0.85
            m.lifetime = rospy.Duration(0.5)
            ma.markers.append(m)
        self.person_radar_pub.publish(ma)

    # -- 公共绘制辅助 --

    def _draw_bboxes(self, img):
        """在图像上绘制YOLO行人bbox"""
        w = img.shape[1]
        for det in self._latest_detections:
            cx = det['center_x']
            cy = det['center_y']
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            cx, cy = int(cx), int(cy)
            half_w = int(det['width'] / 2)
            half_h = int(det['height'] / 2)
            x1 = max(0, cx - half_w)
            y1 = max(0, cy - half_h)
            x2 = min(w - 1, cx + half_w)
            y2 = min(img.shape[0] - 1, cy + half_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
            text_y = max(6, y1 - 6)
            cv2.putText(img, f"Person {det['conf']:.2f}",
                       (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    def _draw_status_bar(self, img, title):
        """在图像顶部绘制状态栏"""
        w = img.shape[1]
        dist_str = f"Dist={self._latest_distance:.2f}m" if self._latest_distance else "Searching..."
        status = f"{title} | {len(self._latest_detections)} person(s) | {dist_str}"
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(img, status, (6, 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self._latest_associated and self.confidence >= self.confidence_threshold:
            cv2.putText(img, "TRACKING", (w - 85, 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # -- 图像叠加 --

    def _draw_overlay(self, header):
        if self.latest_image is None:
            return

        try:
            img = self.latest_image.copy()
        except Exception as e:
            rospy.logwarn_throttle(5, f"[overlay] 图像复制失败: {e}")
            return

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return

        try:
            self._draw_bboxes(img)

            # 原始散点（纯红色，标注坐标）
            for xr, yr, zr, su, sv in self.raw_scatter_points:
                if 0 <= su < w and 0 <= sv < h:
                    cv2.circle(img, (su, sv), 4, (0, 0, 255), -1)
                    cv2.putText(img, f"({xr:.1f},{yr:.1f})", (su + 6, sv - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            self._draw_status_bar(img, "PrintPoint")

            overlay_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            overlay_msg.header = header
            self.printpoint_pub.publish(overlay_msg)

        except Exception as e:
            rospy.logerr_throttle(5, f"[printpoint] 绘制崩溃: {e}")

    def _draw_trajectory_overlay(self, header):
        """绘制轨迹叠加（bbox + KF黄点 + 轨迹线 + 状态栏） → /fusion/printtrajectory"""
        if self.latest_image is None:
            return

        try:
            img = self.latest_image.copy()
        except Exception as e:
            rospy.logwarn_throttle(5, f"[printtrajectory] 图像复制失败: {e}")
            return

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return

        try:
            self._draw_bboxes(img)

            # KF黄点
            if self._latest_associated and self.confidence >= self.confidence_threshold:
                u_a, v_a, Xr, Yr, Zr = self._latest_associated[:5]
                if np.isfinite(u_a) and np.isfinite(v_a):
                    ua, va = int(u_a), int(v_a)
                    if 0 <= ua < w and 0 <= va < h:
                        cv2.circle(img, (ua, va), 8, (0, 255, 255), 2)
                        cv2.circle(img, (ua, va), 3, (0, 255, 255), -1)
                        td = self._latest_distance or 0
                        cv2.putText(img, f"Dist:{td:.2f}m XY:({Xr:.1f},{Yr:.1f})",
                                   (ua + 12, va - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # 轨迹线
            if len(self.trajectory_points) >= 2:
                R_mat = self._cached_R
                t_vec = self._cached_t
                if R_mat is not None and t_vec is not None:
                    draw_pts = self.trajectory_points[-200:]
                    pixel_traj = []
                    for xr, yr, zr in draw_pts:
                        if not np.isfinite(xr) or not np.isfinite(yr):
                            continue
                        xc = R_mat[0, 0] * xr + R_mat[0, 1] * yr + R_mat[0, 2] * zr + t_vec[0]
                        yc = R_mat[1, 0] * xr + R_mat[1, 1] * yr + R_mat[1, 2] * zr + t_vec[1]
                        zc = R_mat[2, 0] * xr + R_mat[2, 1] * yr + R_mat[2, 2] * zr + t_vec[2]
                        if xc > 0.5:
                            ut = self.fx * (-yc / xc) + self.cx
                            vt = self.fy * (-zc / xc) + self.cy
                            if np.isfinite(ut) and np.isfinite(vt) and 0 <= ut < w and 0 <= vt < h:
                                pixel_traj.append((int(ut), int(vt)))
                    n = len(pixel_traj)
                    for i in range(1, n):
                        alpha = 0.3 + 0.7 * (i / n)
                        cv2.line(img, pixel_traj[i - 1], pixel_traj[i],
                                (0, int(255 * alpha), int(255 * alpha)), 2)

            overlay_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            overlay_msg.header = header
            self.printtrajectory_pub.publish(overlay_msg)

        except Exception as e:
            rospy.logerr_throttle(5, f"[printtrajectory] 绘制崩溃: {e}")

    # -- Matplotlib 白底坐标轴画布 --

    def _fig_to_imgmsg(self, fig, header):
        """matplotlib Figure → ROS Image 消息"""
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = buf.reshape(h, w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        ros_img = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        ros_img.header = header
        return ros_img

    def _draw_scatter_canvas(self, header):
        """散点图：白底 + grid + 红色散点 → /fusion/scatter_canvas"""
        fig, ax = plt.subplots(figsize=(6.8, 6.8), facecolor='white')
        ax.set_facecolor('white')

        # 坐标轴: Y(横, ±5m), X(纵, 0~10m)
        # 雷达原点 (Y=0, X=0) 在底部中央
        ax.set_xlim(5.5, -5.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xlabel('Y (m) — 横向', fontsize=11)
        ax.set_ylabel('X (m) — 前方', fontsize=11)
        ax.set_title('Radar Raw Scatter Points', fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal')

        # 雷达原点
        ax.plot(0, 0, 's', color='gray', markersize=8, label='Radar')

        # 散点（红色）
        xs = [yr for _, yr, _, _, _ in self.raw_scatter_points]
        ys = [xr for xr, _, _, _, _ in self.raw_scatter_points]
        if xs:
            ax.scatter(xs, ys, c='red', s=40, edgecolors='darkred',
                       linewidths=0.6, zorder=5, label='Scatter')

        ax.legend(loc='upper right', fontsize=9)

        ros_img = self._fig_to_imgmsg(fig, header)
        self.scatter_canvas_pub.publish(ros_img)

    def _draw_trajectory_canvas(self, header):
        """轨迹图：白底 + grid + 红色轨迹线 + 当前点 → /fusion/trajectory_canvas"""
        fig, ax = plt.subplots(figsize=(6.8, 6.8), facecolor='white')
        ax.set_facecolor('white')

        ax.set_xlim(5.5, -5.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xlabel('Y (m) — 横向', fontsize=11)
        ax.set_ylabel('X (m) — 前方', fontsize=11)
        ax.set_title('KF Trajectory', fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal')

        # 雷达原点
        ax.plot(0, 0, 's', color='gray', markersize=8, label='Radar')

        # 轨迹线（红色渐变粗细）
        pts = self.trajectory_points
        if len(pts) >= 2:
            xs = [yr for _, yr, _ in pts]
            ys = [xr for xr, _, _ in pts]
            # 按新旧分段着色
            n = len(xs)
            chunk = max(1, n // 5)
            colors = plt.cm.Reds(np.linspace(0.4, 1.0, 5))
            for k in range(5):
                i0 = k * chunk
                i1 = min((k + 1) * chunk + 1, n)
                if i0 < i1 - 1:
                    ax.plot(xs[i0:i1], ys[i0:i1], color=colors[k],
                            linewidth=1.5 + 1.5 * (k / 5), zorder=2)
            ax.plot(xs, ys, color='red', alpha=0.25, linewidth=1, zorder=1, label='Trajectory')

        # 当前点（大黄实心圆）
        if self._latest_associated and self.confidence >= self.confidence_threshold:
            _, _, kf_x, kf_y, _, _, _ = self._latest_associated
            ax.plot(kf_y, kf_x, 'o', color='gold', markersize=10,
                    zorder=6, label='Current')
            # 十字准星
            ax.plot(kf_y, kf_x, '+', color='darkorange', markersize=8,
                    markeredgewidth=1.5, zorder=7)

        ax.legend(loc='upper right', fontsize=9)

        ros_img = self._fig_to_imgmsg(fig, header)
        self.trajectory_canvas_pub.publish(ros_img)

    # -- 原始散点发布 --

    def _publish_raw_scatter(self, header):
        """发布原始关联散点到RViz（radar_link坐标系，不经滤波）"""
        ma = MarkerArray()
        # 先发一个DELETEALL清除旧marker（防止点数减少时残留）
        del_m = Marker()
        del_m.ns = 'raw_scatter'
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)
        n = len(self.raw_scatter_points)
        for i, (xr, yr, zr, _, _) in enumerate(self.raw_scatter_points):
            m = Marker()
            m.header = header
            m.header.frame_id = 'radar_link'
            m.ns = 'raw_scatter'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = xr
            m.pose.position.y = yr
            m.pose.position.z = zr
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.9
            m.lifetime = rospy.Duration(0)  # 永久显示
            ma.markers.append(m)
            # 坐标文字标注
            tm = Marker()
            tm.header = header
            tm.header.frame_id = 'radar_link'
            tm.ns = 'raw_scatter_text'
            tm.id = i
            tm.type = Marker.TEXT_VIEW_FACING
            tm.action = Marker.ADD
            tm.pose.position.x = xr
            tm.pose.position.y = yr
            tm.pose.position.z = zr + 0.3
            tm.pose.orientation.w = 1.0
            tm.scale.z = 0.3
            tm.color.r = 1.0; tm.color.g = 1.0; tm.color.b = 1.0; tm.color.a = 0.9
            tm.text = f"({xr:.1f},{yr:.1f})"
            tm.lifetime = rospy.Duration(0)
            ma.markers.append(tm)
        self.raw_scatter_pub.publish(ma)

    # -- 辅助方法 --

    def _clear_state(self, full_reset=False):
        """清空缓存状态。full_reset=True 时同时重置跟踪和轨迹。"""
        self._latest_detections = []
        self._latest_associated = None
        self._latest_distance = None
        self._innov_window.clear()
        if full_reset:
            self.kf.reset()
            self.last_position_radar = None
            self.trajectory_points = []
            self.raw_scatter_points = []
            self._last_scatter_time = None
            self.confidence = 0
            self.lost_count = 0

    def _publish_cleared(self, header):
        ma = MarkerArray()
        m = Marker()
        m.ns = 'person_radar'
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.person_radar_pub.publish(ma)

    def _parse_yolo(self, msg):
        try:
            data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1').flatten()
        except Exception:
            return []
        if len(data) == 0:
            return []
        dets = []
        for i in range(len(data) // 5):
            idx = i * 5
            dets.append({
                'center_x': float(data[idx]),
                'center_y': float(data[idx + 1]),
                'width': float(data[idx + 2]),
                'height': float(data[idx + 3]),
                'conf': float(data[idx + 4])
            })
        return dets

    def _parse_radar(self, msg):
        field_names_set = {f.name for f in msg.fields}
        has_vy = 'vy' in field_names_set
        pts = []
        for pt in pc2.read_points(msg, skip_nans=True):
            Xr, Yr, Zr = pt[0], pt[1], pt[2]
            if np.sqrt(Xr**2 + Yr**2) < self.max_radar_dist:
                Vx = pt[3] if len(pt) > 3 else 0.0
                Vy = pt[4] if has_vy and len(pt) > 4 else 0.0
                pts.append((Xr, Yr, Zr, Vx, Vy))
        return pts

    def _get_tf(self, stamp):
        try:
            trans = self.tf_buffer.lookup_transform(
                'camera_link', 'radar_link', stamp, rospy.Duration(0.05))
        except Exception:
            return None
        t = np.array([trans.transform.translation.x,
                      trans.transform.translation.y,
                      trans.transform.translation.z])
        q = trans.transform.rotation
        R = self._q2m(q.x, q.y, q.z, q.w)
        return R, t

    @staticmethod
    def _q2m(x, y, z, w):
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2*y**2 - 2*z**2
        R[0, 1] = 2*x*y - 2*z*w
        R[0, 2] = 2*x*z + 2*y*w
        R[1, 0] = 2*x*y + 2*z*w
        R[1, 1] = 1 - 2*x**2 - 2*z**2
        R[1, 2] = 2*y*z - 2*x*w
        R[2, 0] = 2*x*z - 2*y*w
        R[2, 1] = 2*y*z + 2*x*w
        R[2, 2] = 1 - 2*x**2 - 2*y**2
        return R

    def _handle_lost(self):
        if self.lost_count > self.max_lost:
            self.kf.reset()
            self.last_position_radar = None
            self.confidence = 0
            self._innov_window.clear()
        elif self.confidence > 0:
            self.confidence = max(0, self.confidence - 1)

    def _associate(self, detections, projected):
        """关联行人bbox与雷达投影点，自适应阈值+KF预测辅助"""
        if not projected:
            return None

        best_global = best_global_score = None
        kf_pred = self.kf.get_predicted_position(dt=self._current_dt)

        for det in detections:
            cx_d, cy_d = det['center_x'], det['center_y']
            half_w, half_h = det['width'] / 2.0, det['height'] / 2.0
            bbox_area = det['width'] * det['height']
            adaptive_threshold = self.association_threshold + max(0, 30000 - bbox_area) * 0.005
            best_det = best_det_score = None

            for Xr, Yr, Zr, Xc, Yc, Zc, u, v, Vxp, Vyp in projected:
                px_d = np.sqrt((u - cx_d)**2 + (v - cy_d)**2)
                if px_d > adaptive_threshold:
                    continue
                in_bbox = (abs(u - cx_d) < half_w + 5 and abs(v - cy_d) < half_h + 5)
                jump_d = 0.0
                if self.last_position_radar is not None:
                    lx, ly, _ = self.last_position_radar
                    jump_d = np.sqrt((Xr - lx)**2 + (Yr - ly)**2)
                    dist_pt = np.sqrt(Xr**2 + Yr**2)
                    effective_assoc_jump = self.max_jump_dist + max(0, (dist_pt - 5.0) * 0.3) + 2
                    if jump_d > effective_assoc_jump:
                        continue
                kf_d = 0.0
                if kf_pred is not None and self.kf.initialized:
                    kf_d = np.sqrt((Xr - kf_pred[0])**2 + (Yr - kf_pred[1])**2)
                score = px_d + (-300 if in_bbox else 0) + jump_d * 2 + kf_d * 5
                if best_det_score is None or score < best_det_score:
                    best_det_score = score
                    best_det = (Xr, Yr, Zr, Xc, Yc, Zc, u, v, Vxp, Vyp)

            if best_det and (best_global_score is None or best_det_score < best_global_score):
                best_global_score, best_global = best_det_score, best_det

        return best_global

    def _publish_person(self, header, cam_pt):
        Xc, Yc, Zc = cam_pt
        ps = PointStamped()
        ps.header = header
        ps.header.frame_id = 'camera_link'
        ps.point.x, ps.point.y, ps.point.z = Xc, Yc, Zc
        self.person_pub.publish(ps)

    def _update_trajectory(self, header, radar_pos):
        xr, yr, zr = radar_pos
        self.trajectory_points.append((xr, yr, zr))
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points = self.trajectory_points[-self.max_trajectory_points:]
        n = len(self.trajectory_points)

        cm = Marker()
        cm.header = header
        cm.header.frame_id = 'radar_link'
        cm.ns = 'person_current'
        cm.id = 0
        cm.type = Marker.SPHERE
        cm.action = Marker.ADD
        cm.pose.position.x, cm.pose.position.y, cm.pose.position.z = xr, yr, zr
        cm.pose.orientation.w = 1.0
        cm.scale.x = cm.scale.y = cm.scale.z = 0.5
        cm.color.g = 1.0
        cm.color.a = 1.0
        cm.lifetime = rospy.Duration(2.0)
        self.current_pos_pub.publish(cm)

        tm = Marker()
        tm.header = header
        tm.header.frame_id = 'radar_link'
        tm.ns = 'trajectory'
        tm.id = self.trajectory_id
        tm.type = Marker.LINE_STRIP
        tm.action = Marker.ADD
        tm.scale.x = 0.1
        tm.color.r = 1.0
        tm.color.a = 0.9
        for x, y, z in self.trajectory_points:
            p = Point(); p.x, p.y, p.z = x, y, z
            tm.points.append(p)
        if n > 1:
            tm.colors = [ColorRGBA(1, 0, 0, 0.2 + 0.8*i/n) for i in range(n)]
        self.trajectory_pub.publish(tm)

        self.trajectory_count += 1
        if self.trajectory_count % 100 == 0:
            self.trajectory_id += 1


if __name__ == '__main__':
    try:
        node = FusionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
