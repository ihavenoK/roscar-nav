#!/usr/bin/env python3
# coding=utf-8
"""
毫米波雷达 + YOLO 行人动态障碍物节点

流程:
  1. YOLO 检测行人 → pixel bbox
  2. 毫米波雷达点云 → TF 投影到像素
  3. 每个 YOLO bbox 内的雷达点 → 聚类成 30cm 圆形区域
  4. 发布 PointCloud2 → 局部代价地图自动避障

原理:
  costmap 的 obstacle_layer 订阅 /person_obstacle_cloud，
  move_base 收到后自动修改局部代价地图 → 自动规划避障路线。
  全程不需要人为调用 move_base 的 recovery。
"""

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs  # 注册 PointStamped 的 TF 变换（camera_link→odom_combined）
import message_filters
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


class PedestrianObstacleNode:
    """YOLO + 雷达 → 行人障碍物点云 → costmap 自动避障"""

    def __init__(self):
        rospy.init_node("pedestrian_obstacle")

        # ---- 相机内参 ----
        self.fx = rospy.get_param("~fx", 400.0)
        self.fy = rospy.get_param("~fy", 400.0)
        self.cx = rospy.get_param("~cx", 320.0)
        self.cy = rospy.get_param("~cy", 240.0)

        # ---- 聚类/碰撞参数 ----
        self.conf_min         = rospy.get_param("~conf_min", 0.2)
        self.max_time_diff    = rospy.get_param("~max_time_diff", 0.5)
        self.cluster_radius   = rospy.get_param("~cluster_radius", 0.5)  # 每bbox生成0.5m圆，绕开走
        self.min_radar_points = rospy.get_param("~min_radar_points", 1)  # bbox内至少几个雷达点
        self.obstacle_lifetime = rospy.get_param("~obstacle_lifetime", 0.5)  # costmap保留时长

        # ---- 时序追踪参数（解决 YOLO 置信度波动导致闪烁）----
        self.tracked_persons = {}  # {tid: {"center":np, "bbox":(x1,y1,w,h), "conf":f, "last_seen":Time, "miss_count":int}}
        self.next_track_id = 0
        self.track_max_dist  = rospy.get_param("~track_max_dist", 1.0)   # 同一行人匹配距离阈值(m)
        self.max_miss_frames = rospy.get_param("~max_miss_frames", 10)   # 允许连续丢帧数(10Hz下10帧=1.0s)
        # 近距离自适应保持：越近保持越久，防止短暂漏检导致冲撞
        self.near_hold_dist       = rospy.get_param("~near_hold_dist", 1.5)       # 触发近距离保持的距离阈值(m)
        self.near_max_miss_frames = rospy.get_param("~near_max_miss_frames", 50)  # 近距离保持帧数(5s@10Hz)

        # ---- 雷达过滤参数 ----
        # 距离门
        self.radar_max_dist  = rospy.get_param("~radar_max_dist", 10.0)
        # RCS 门
        self.rcs_max         = rospy.get_param("~rcs_max", 15.0)
        # 行人平均身高(m)，雷达无回波时用 YOLO bbox 高度估距兜底
        self.person_height   = rospy.get_param("~person_height", 1.7)
        # 固定世界坐标系（不随车移动，静止行人在此坐标系中位置不变）
        self.odom_frame      = rospy.get_param("~odom_frame", "odom_combined")

        # ---- 时间同步订阅（按实际时间戳精确匹配雷达+YOLO） ----
        self.radar_sub = message_filters.Subscriber("/radar/pointcloud", PointCloud2)
        self.yolo_sub  = message_filters.Subscriber("/yolo/person_detections", Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.radar_sub, self.yolo_sub],
            queue_size=30,
            slop=self.max_time_diff)  # 时间窗口
        self.sync.registerCallback(self.synced_callback)

        # ---- 发布 ----
        # 行人障碍物点云 → costmap obstacle_layer 订阅
        self.pub_cloud = rospy.Publisher(
            "/person_obstacle_cloud", PointCloud2, queue_size=5)

        # 调试：行人位置
        self.pub_position   = rospy.Publisher("/fusion/person_position", PointStamped, queue_size=5)
        self.pub_markers    = rospy.Publisher("/fusion/person_markers", MarkerArray, queue_size=5)

        # 标记生命周期跟踪
        self._active_marker_ids = set()
        self._marker_lifetime = 0.5

        # ---- TF ----
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("PedestrianObstacleNode ready. cluster=%.1fm lifetime=%.1fs",
                      self.cluster_radius, self.obstacle_lifetime)

    # -- 主处理（时间同步回调） --

    def synced_callback(self, radar_msg, yolo_msg):
        """ApproximateTimeSynchronizer 回调：雷达+YOLO 已按实际时间戳精确匹配"""

        # ---- 解析 ----
        yolo_boxes = self._parse_yolo(yolo_msg)
        if not yolo_boxes:
            # YOLO 漏检时仍维护近距离 tracker，防止障碍物点云中断
            return self._handle_no_yolo(yolo_msg.header, radar_msg.header.stamp)

        radar_pts = self._parse_radar(radar_msg)
        if not radar_pts:
            # 雷达无点 → 无法融合，但仍需维护 tracker 并发布历史位置
            return self._handle_no_yolo(yolo_msg.header, radar_msg.header.stamp)

        # ---- TF ----
        R, t = self._get_tf(radar_msg.header.stamp)
        if R is None:
            # 雷达→相机 TF 不可用，回落 tracker 维护
            return self._handle_no_yolo(yolo_msg.header, radar_msg.header.stamp)

        # ---- 对每个 YOLO bbox 聚类 ----
        person_centroids = []  # [(cam_pt, bbox, conf, cluster_id)]
        person_id = 0

        for box in yolo_boxes:
            cx_b, cy_b, w, h, conf = box
            x1 = int(cx_b - w / 2)
            y1 = int(cy_b - h / 2)
            x2 = int(cx_b + w / 2)
            y2 = int(cy_b + h / 2)

            # bbox 内的雷达点（已在 _parse_radar 过滤 dist>3m 与 高RCS 点）
            in_box = []
            for pt in radar_pts:
                uv = self._radar_to_pixel(pt, R, t)
                if uv is None:
                    continue
                u, v = uv
                if x1 <= u <= x2 and y1 <= v <= y2:
                    in_box.append(pt)

            if len(in_box) >= self.min_radar_points:
                # 转到 camera 坐标系
                cam_pts = np.array([
                    R @ np.array([pt[0], pt[1], pt[2]]) + t for pt in in_box
                ])
                # 取距离相机最近的点作为行人位置（防止残留杂波平均拉偏）
                # 旧方案 np.mean 会把人与墙的点平均，导致距离偏大
                dists = np.linalg.norm(cam_pts, axis=1)
                center_cam = cam_pts[int(np.argmin(dists))]
            else:
                # 雷达未检测到该行人（RCS 过低或超出有效距离）
                # 兜底：用 YOLO bbox 高度按针孔模型估距，
                # 并利用 bbox 水平偏移反投影估算 Y 方向（避免僵死在正前方 y=0）
                if h <= 0:
                    continue
                bbox_dist = self.fy * self.person_height / h
                # 反投影公式（与 _radar_to_pixel 一致）：
                #   u = cx + fx * (-Y / X)  →  Y = -(u - cx) * X / fx
                est_y = -(cx_b - self.cx) * bbox_dist / self.fx
                # Z 近似相机高度（行人中心约与相机等高），保持 0
                center_cam = np.array([bbox_dist, est_y, 0.0])
                rospy.logdebug("bbox估距: 人 %.2fm, y=%.2f (无雷达点)", bbox_dist, est_y)

            # 记录此人（点簇稍后由追踪器统一生成）
            person_centroids.append((center_cam, (x1, y1, x2 - x1, y2 - y1), conf, person_id))
            person_id += 1

        # ---- 时序追踪平滑：匹配历史 tracker，短时丢失仍保持位置 ----
        person_centroids = self._match_trackers(person_centroids, yolo_msg.header.stamp)

        # ---- 生成障碍物点云（camera_link → odom_combined 变换后发布到固定坐标系）----
        # 只处理 3m 内的行人，远处的不纳入避障
        cluster_points = []
        for cam_pt, bbox, conf, pid in person_centroids:
            if np.linalg.norm(cam_pt) > 3.0:
                continue
            cluster_pts = self._generate_cluster(cam_pt, self.cluster_radius)
            cluster_points.extend(cluster_pts)

        # ---- 发布障碍物点云（用雷达时戳做 TF，转 odom_combined 后钉死在固定位置）----
        if cluster_points:
            self._publish_cloud(cluster_points, radar_msg.header.stamp)
        else:
            self._publish_cloud([], radar_msg.header.stamp)
            self._publish_markers([], yolo_msg.header)
            return

        # ---- 发布 RViz 行人标记 ----
        self._publish_markers(person_centroids, yolo_msg.header)

        # ---- 发布最近行人位置 + 叠加图 ----
        # 取最近的人
        best = min(person_centroids, key=lambda x: np.linalg.norm(x[0]))
        cam_pt, bbox, conf, _ = best
        p = PointStamped()
        p.header = yolo_msg.header
        p.header.frame_id = "camera_link"
        p.point.x = cam_pt[0]
        p.point.y = cam_pt[1]
        p.point.z = cam_pt[2]
        self.pub_position.publish(p)

    # -- 障碍物点云发布（odom_combined 固定坐标系） --

    def _publish_markers(self, person_centroids, header):
        """发布 RViz MarkerArray：每个行人一个半透明球体 + 距离标签

        优：odom_combined 固定坐标系（人不随车动）
        次：camera_link 兜底（至少能看到标记）
        """
        markers = MarkerArray()
        new_ids = set()

        # 尝试将 marker 位置变换到 odom_combined，让人在 RViz 中也固定
        marker_frame = self.odom_frame
        try_transform = True
        try:
            # 预检 TF 是否可用（与 _transform_to_odom 共用 1.0s 超时）
            can = self.tf_buffer.can_transform(
                self.odom_frame, "camera_link", header.stamp,
                rospy.Duration(1.0))
            if not can:
                try_transform = False
                marker_frame = "camera_link"
                rospy.logwarn_throttle(2.0,
                    "TF camera_link→%s at stamp=%.3f not available for markers, fallback camera_link",
                    self.odom_frame, header.stamp.to_sec())
        except Exception as e:
            try_transform = False
            marker_frame = "camera_link"
            rospy.logwarn_throttle(2.0,
                "TF camera_link→%s precheck failed for markers: %s",
                self.odom_frame, str(e)[:120])

        for cam_pt, bbox, conf, pid in person_centroids:
            mid = pid
            d = np.linalg.norm(cam_pt)

            # 变换到 odom_combined（若可用）
            pos_x, pos_y = cam_pt[0], cam_pt[1]
            pos_z = cam_pt[2]
            if try_transform:
                try:
                    pt = PointStamped()
                    pt.header.stamp = header.stamp
                    pt.header.frame_id = "camera_link"
                    pt.point.x = cam_pt[0]
                    pt.point.y = cam_pt[1]
                    pt.point.z = cam_pt[2]
                    pt_odom = self.tf_buffer.transform(
                        pt, self.odom_frame, rospy.Duration(1.0))
                    pos_x, pos_y, pos_z = pt_odom.point.x, pt_odom.point.y, pt_odom.point.z
                except Exception:
                    pass

            # 球体标记
            sphere = Marker()
            sphere.header.frame_id = marker_frame
            sphere.header.stamp = header.stamp
            sphere.ns = "person"
            sphere.id = mid
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = pos_x
            sphere.pose.position.y = pos_y
            sphere.pose.position.z = pos_z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = self.cluster_radius * 2
            sphere.scale.y = self.cluster_radius * 2
            sphere.scale.z = self.cluster_radius * 2
            sphere.color.r = 1.0
            sphere.color.g = 0.3
            sphere.color.b = 0.2
            sphere.color.a = 0.6
            sphere.lifetime = rospy.Duration.from_sec(self._marker_lifetime)
            markers.markers.append(sphere)
            new_ids.add(("person", mid))

            # 距离文字
            text = Marker()
            text.header.frame_id = marker_frame
            text.header.stamp = header.stamp
            text.ns = "person_text"
            text.id = mid
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = pos_x
            text.pose.position.y = pos_y
            text.pose.position.z = pos_z + 0.6
            text.scale.z = 0.3
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"{d:.1f}m"
            text.lifetime = rospy.Duration.from_sec(self._marker_lifetime)
            markers.markers.append(text)
            new_ids.add(("person_text", mid))

        # 清除不再出现的旧标记
        stale_ids = self._active_marker_ids - new_ids
        for ns, mid in stale_ids:
            dm = Marker()
            dm.header.frame_id = marker_frame
            dm.header.stamp = header.stamp
            dm.ns = ns
            dm.id = mid
            dm.action = Marker.DELETE
            markers.markers.append(dm)

        self._active_marker_ids = new_ids
        self.pub_markers.publish(markers)

    # -- 障碍物点云发布（odom_combined 固定坐标系） --

    def _generate_cluster(self, center, radius):
        """以 center 为中心，生成半径 radius 的球形点簇（返回 world 坐标列表）"""
        pts = []
        steps = 4  # 球壳层数 (6→4, 省CPU)
        for r in np.linspace(0, radius, steps):
            n_pts = max(4, int(10 * (r / radius)))
            for i in range(n_pts):
                theta = 2 * np.pi * i / n_pts
                # 水平圆面上撒点
                pts.append([center[0], center[1] + r * np.cos(theta),
                            center[2] + r * np.sin(theta)])
                pts.append([center[0] + r * np.cos(theta), center[1],
                            center[2] + r * np.sin(theta)])
        # 中心点加强
        for _ in range(5):
            pts.append([center[0] + np.random.uniform(-0.05, 0.05),
                        center[1] + np.random.uniform(-0.05, 0.05),
                        center[2] + np.random.uniform(-0.05, 0.05)])
        return pts

    def _publish_cloud(self, points, radar_stamp):
        """发布障碍物 PointCloud2。

        优：camera_link→odom_combined 固定坐标系（人不随车动）
        劣：TF 不可用时回退 camera_link 发布（至少数据不丢失）

        注意：绝不能用 rospy.Time(0) 做 TF，那会把车移动距离错加给人，
        导致静止人也跟着车跑。
        """
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        if not points:
            cloud_header = Header()
            cloud_header.stamp = radar_stamp
            cloud_header.frame_id = self.odom_frame
            self.pub_cloud.publish(pc2.create_cloud(cloud_header, fields, []))
            return

        # 唯一路径：用雷达时戳精确查 camera_link→odom_combined TF
        odom_points = self._transform_to_odom(points, radar_stamp)
        if odom_points is not None:
            cloud_header = Header()
            cloud_header.stamp = radar_stamp
            cloud_header.frame_id = self.odom_frame
            self.pub_cloud.publish(pc2.create_cloud(cloud_header, fields, odom_points))
            return

        # TF 不可用 → 兜底仍发 odom_combined（保持坐标系一致，避免 costmap 因
        # sensor_frame 不匹配而 worldToMap 失败打出 "out of map bounds" WARN）
        rospy.logwarn_throttle(2.0,
            "TF camera_link→%s 在 stamp=%.3f 不可用 → 回退 odom_combined 发布（坐标近似）",
            self.odom_frame, radar_stamp.to_sec())
        cloud_header = Header()
        cloud_header.stamp = radar_stamp
        cloud_header.frame_id = self.odom_frame
        self.pub_cloud.publish(pc2.create_cloud(cloud_header, fields, points))

    def _transform_to_odom(self, points, stamp):
        """camera_link → odom_combined 批量变换。

        成功返回 odom 坐标列表，失败返回 None。
        stamp 必须是观察时刻的精确时戳，不能用 rospy.Time(0)。
        """
        # 先检查 TF 是否可用，避免逐点循环中反复失败
        try:
            can = self.tf_buffer.can_transform(
                self.odom_frame, "camera_link", stamp, rospy.Duration(1.0))
            if not can:
                rospy.logwarn_throttle(2.0,
                    "TF camera_link→%s at stamp=%.3f not available",
                    self.odom_frame, stamp.to_sec())
                return None
        except Exception:
            pass

        try:
            odom_points = []
            for pt_xyz in points:
                pt = PointStamped()
                pt.header.stamp = stamp
                pt.header.frame_id = "camera_link"
                pt.point.x = pt_xyz[0]
                pt.point.y = pt_xyz[1]
                pt.point.z = pt_xyz[2]
                pt_odom = self.tf_buffer.transform(
                    pt, self.odom_frame, rospy.Duration(1.0))
                odom_points.append(
                    [pt_odom.point.x, pt_odom.point.y, pt_odom.point.z])
            return odom_points
        except Exception as e:
            rospy.logwarn_throttle(2.0,
                "TF camera_link→%s failed: %s", self.odom_frame, str(e)[:120])
            return None

    def _handle_no_yolo(self, yolo_header, radar_stamp):
        """YOLO 无检测时的兜底处理。

        近距离 tracker 继续发布最后已知位置的点云，
        防止 YOLO 短暂漏检（半身/暗光）导致障碍物从 costmap 消失。
        远距离 tracker 不发布幽灵障碍物，交给激光 /scan 兜底。
        """
        # 即使无 YOLO 检测，也要维护 tracker 的 miss_count
        near_trackers = self._match_trackers([], yolo_header.stamp)

        if not self.tracked_persons:
            self._publish_cloud([], radar_stamp)
            self._publish_markers([], yolo_header)
            return

        # 只对近距离且未超丢失阈值的 tracker 继续发布点云
        cluster_points = []
        near_detections = []
        for tid, trk in self.tracked_persons.items():
            last_dist = float(np.linalg.norm(trk["center"]))
            trk["last_dist"] = last_dist
            max_miss = self._max_miss_for(trk)
            if last_dist < self.near_hold_dist and trk["miss_count"] < max_miss:
                cluster_pts = self._generate_cluster(trk["center"], self.cluster_radius)
                cluster_points.extend(cluster_pts)
                near_detections.append((trk["center"], trk["bbox"], trk["conf"], tid))

        if cluster_points:
            self._publish_cloud(cluster_points, radar_stamp)
            self._publish_markers(near_detections, yolo_header)
        else:
            self._publish_cloud([], radar_stamp)
            self._publish_markers([], yolo_header)

    def _max_miss_for(self, trk):
        """根据最后检测距离返回允许丢帧数。近距离延长保持，防止 YOLO 漏检导致冲撞。"""
        if trk.get("last_dist", 99.0) < self.near_hold_dist:
            return self.near_max_miss_frames
        return self.max_miss_frames

    def _match_trackers(self, detections, stamp):
        """时序行人追踪：将本帧检测匹配到历史 tracker，短时丢失仍保持。
        
        Args:
            detections: list of (cam_pt, bbox, conf, pid)
            stamp: 当前帧时间戳
        
        Returns:
            平滑后的检测列表 (含历史保持的"虚"tracker)，格式同 detections
        """
        used_track_ids = set()

        for cam_pt, bbox, conf, pid in detections:
            # 寻找最近的已存在 tracker（2D 平面距离匹配）
            best_tid = None
            best_dist = self.track_max_dist
            for tid, trk in self.tracked_persons.items():
                if tid in used_track_ids:
                    continue
                d = np.linalg.norm(trk["center"][:2] - cam_pt[:2])
                if d < best_dist:
                    best_dist = d
                    best_tid = tid

            if best_tid is not None:
                # 匹配成功：更新已有 tracker
                trk = self.tracked_persons[best_tid]
                trk["center"] = cam_pt
                trk["bbox"] = bbox
                trk["conf"] = conf
                trk["last_seen"] = stamp
                trk["miss_count"] = 0
                trk["last_dist"] = float(np.linalg.norm(cam_pt))
                used_track_ids.add(best_tid)
            else:
                # 新人：创建 tracker
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracked_persons[tid] = {
                    "center": cam_pt,
                    "bbox": bbox,
                    "conf": conf,
                    "last_seen": stamp,
                    "miss_count": 0,
                    "last_dist": float(np.linalg.norm(cam_pt)),
                }
                used_track_ids.add(tid)

        # 本帧未匹配到的旧 tracker：累计丢帧计数
        for tid, trk in self.tracked_persons.items():
            if tid not in used_track_ids:
                trk["miss_count"] += 1

        # 清除连续丢帧超过阈值的 tracker（近距离阈值自适应提高）
        stale_ids = [tid for tid, trk in self.tracked_persons.items()
                     if trk["miss_count"] >= self._max_miss_for(trk)]
        for tid in stale_ids:
            del self.tracked_persons[tid]

        # 返回所有活跃 tracker 的检测记录（含历史保持的"虚"检测）
        result = []
        for tid, trk in self.tracked_persons.items():
            if trk["miss_count"] < self._max_miss_for(trk):
                result.append((trk["center"], trk["bbox"], trk["conf"], tid))
        return result

    # -- 解析 --

    def _parse_yolo(self, msg):
        if msg.encoding != "32FC1":
            return []
        det = np.frombuffer(msg.data, dtype=np.float32)
        if det.size == 0 or det.size < 5:
            return []
        boxes = det.reshape(-1, 5)
        return [b for b in boxes if b[4] >= self.conf_min]

    def _parse_radar(self, msg):
        pts = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "vx", "vy", "dist", "rcs"),
                                  skip_nans=True):
            # 索引: 0=x 1=y 2=z 3=vx 4=vy 5=dist 6=rcs
            # 距离门：丢弃 >3m 的远距离杂波（身后墙壁等）
            if p[5] > self.radar_max_dist:
                continue
            # RCS 门：丢弃高反射目标（金属墙、车辆 >5 dBsm）
            if p[6] > self.rcs_max:
                continue
            pts.append(p)
        return pts

    # -- TF / 投影 --

    def _get_tf(self, stamp):
        try:
            trans = self.tf_buffer.lookup_transform(
                "camera_link", "radar_link", stamp, rospy.Duration(0.1))
            t = trans.transform.translation
            q = trans.transform.rotation
            R = self._q2m(q.x, q.y, q.z, q.w)
            T = np.array([t.x, t.y, t.z])
            return R, T
        except Exception:
            return None, None

    def _radar_to_pixel(self, radar_pt, R, t):
        """radar_link → camera_link → (u, v)"""
        p = R @ np.array([radar_pt[0], radar_pt[1], radar_pt[2]]) + t
        X = p[0]
        if X <= 0:
            return None
        u = int(self.cx + self.fx * (-p[1] / X))
        v = int(self.cy + self.fy * (-p[2] / X))
        return u, v

    @staticmethod
    def _q2m(x, y, z, w):
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])


if __name__ == "__main__":
    try:
        node = PedestrianObstacleNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
