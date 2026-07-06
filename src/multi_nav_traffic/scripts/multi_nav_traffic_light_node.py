#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-point navigation with dynamic model switching.

Architecture (按需加载, 节省GPU显存):
  - NAVIGATING 状态: 加载 yolov8n.pt → 检测行人 → 发布 /yolo/person_detections
                     → fusion_person_detect 融合雷达 → costmap 自动避障
  - 单数WP到达后 (红绿灯前1.5m): 停车 → 卸载 yolov8n.pt → 加载 best_new.pt (~3-8s)
                     → RED_LIGHT_CHECK
  - RED_LIGHT_CHECK: 检测绿灯 → 卸载 best_new.pt → 加载 yolov8n.pt (~3-8s)
                     → 继续下一个WP
  - 同一时刻只加载一个模型, 节省约一半GPU显存

数据流:
  /usb_cam/image_raw
    ↓ (单YOLO, 模型按状态切换)
    ├── NAVIGATING:      person bbox → /yolo/person_detections → 雷达融合避障
    └── RED_LIGHT_CHECK: red/green/yellow → 等绿灯

Author: 蔡博涵
"""

import os
import rospy
import tf
import math
import time
import gc
import numpy as np
import cv2
import actionlib
import collections
import threading

from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
from tf2_geometry_msgs import do_transform_point

from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import PointStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

try:
    import torch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # Jetson: 关闭节省显存
except ImportError:
    torch = None

from ultralytics import YOLO


# --
#  Gamma correction utility
# --
def build_gamma_table(gamma):
    """gamma > 1 darkens image (outdoor bright light), gamma < 1 brightens."""
    if abs(gamma - 1.0) < 1e-6:
        return None
    inv_gamma = 1.0 / gamma
    return np.array([(i / 255.0) ** inv_gamma * 255
                     for i in range(256)]).astype("uint8")


# --
#  Main node class
# --
class MultiNavTrafficLightNode:
    """Multi-point navigation with dynamic YOLO model switching."""

    # --- Navigation state machine ---
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    NAVIGATING = "NAVIGATING"              # 行走中: yolov8n.pt 检测行人
    RED_LIGHT_CHECK = "RED_LIGHT_CHECK"    # 红绿灯检测: best_new.pt
    DONE = "DONE"

    # --- YOLO class mapping ---
    # yolov8n.pt (COCO): class 0 = person
    PERSON_CLASS_ID = 0
    # best_new.pt (3类): 0=green, 1=red, 2=yellow
    TL_CLASS_NAMES = {0: "green", 1: "red", 2: "yellow"}

    # --- OpenCV -> ROS coordinate rotation ---
    R_CV_TO_ROS = np.array([[0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]], dtype=np.float64)

    def __init__(self):
        rospy.init_node("multi_nav_traffic_light_node")

        # ---- ROS parameters ----
        self.person_model_path = rospy.get_param("~person_model_path", "yolov8n.pt")
        self.tl_model_path     = rospy.get_param("~tl_model_path")
        self.gamma              = rospy.get_param("~gamma", 1.5)
        self.camera_frame       = rospy.get_param("~camera_frame", "camera_link")
        self.laser_frame        = rospy.get_param("~laser_frame", "laser")
        self.robot_frame        = rospy.get_param("~robot_frame", "base_footprint")
        self.stop_distance      = rospy.get_param("~stop_distance", 3.0)
        self.hold_distance      = rospy.get_param("~hold_distance", 1.5)
        self.angle_tolerance    = math.radians(rospy.get_param("~angle_tolerance", 2.0))
        self.green_confirm_n    = rospy.get_param("~green_confirm_frames", 3)
        self.red_wait_timeout   = rospy.get_param("~red_wait_timeout", 30.0)

        # ---- Model switching params ----
        self.model_switch_timeout = rospy.get_param("~model_switch_timeout", 15.0)
        self.person_conf_thresh   = rospy.get_param("~person_conf_threshold", 0.2)
        self.tl_conf_thresh       = rospy.get_param("~tl_conf_threshold", 0.5)

        # ---- 原地旋转扫描红绿灯参数 ----
        self.tl_scan_angle   = math.radians(rospy.get_param("~tl_scan_angle", 30.0))   # 左右各30度
        self.tl_scan_vel     = rospy.get_param("~tl_scan_angular_vel", 0.3)            # 扫描角速度 rad/s

        # ---- Distance filter (for traffic light distance via LiDAR) ----
        self.dist_filter_window = rospy.get_param("~distance_filter_window", 5)

        self.show_window = rospy.get_param("~show_detection_window", False)

        # ---- Frame rate throttling (避免帧积压导致延迟累积) ----
        self._last_infer_time = 0.0
        self._min_infer_interval = rospy.get_param("~min_infer_interval", 0.07)  # ~14fps max

        # ---- Pre-computed static laser→robot transform (for fast LiDAR distance lookup) ----
        self._laser_tf_cached = False
        self._laser_yaw_offset = 0.0   # yaw of laser_frame relative to robot_frame
        self._laser_tx = 0.0           # translation x
        self._laser_ty = 0.0           # translation y

        # Camera intrinsics (for person bbox encoding & TL angle estimation)
        self.fx = rospy.get_param("~fx", 400.0)
        self.fy = rospy.get_param("~fy", 400.0)
        self.cx = rospy.get_param("~cx", 320.0)
        self.cy = rospy.get_param("~cy", 240.0)

        # ---- Gamma lookup table ----
        self.gamma_table = build_gamma_table(self.gamma)

        # ---- TF ----
        self.tf_listener = tf.TransformListener()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ---- YOLO models (按需加载, 切换时卸载不用的, 节省 GPU 显存) ----
        self.bridge = CvBridge()
        # 不再预加载 — 启动后根据状态按需加载
        self.active_model = None    # 当前活跃 YOLO 对象
        self.active_mode  = None    # "person" or "tl"
        self._use_half    = False   # 是否使用 FP16 推理（Jetson 上由 warmup 决定）

        # 提前占位，防止 publisher 创建前 subscriber 回调中使用时报 AttributeError
        self.pub_person_det = None
        self.pub_status     = None
        self.pub_distance   = None
        self.pub_nav        = None
        self.pub_model      = None
        self.pub_cmd_vel    = None
        self.pub_waypoints  = None

        # ---- Detection state ----
        self.detected_class  = None          # "red"/"green"/"yellow" (TL mode)
        self.target_angle    = None          # angle to traffic light (rad)
        self.light_distance  = float("inf")
        self.latest_frame    = None

        # ---- EMA distance filter ----
        self._dist_buffer = collections.deque(maxlen=self.dist_filter_window)
        self._dist_filtered = float("inf")

        # ---- Waypoint state ----
        self.waypoints       = []
        self.current_wp_idx  = 0
        self.state           = self.IDLE

        # ---- move_base action client ----
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server ...")
        if not self.move_base_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("move_base action server not available after 10s")
        else:
            rospy.loginfo("move_base action server connected.")

        # ---- Model switching state ----
        self._switch_start_time = None
        self._switch_target_mode = None
        self._green_count = 0
        self._red_seen_count = 0
        self._red_wait_start = None
        self._model_corrupted = False   # OOM 后标记，触发自动重载
        self._model_switch_lock = threading.Lock()  # 防止模型切换竞态

        # ---- Goal retry cooldown (防止 ABORTED/REJECTED 时 10Hz 重发塞爆 move_base) ----
        self._retry_cooldown = 2.0          # 重试冷却 2s
        self._last_retry_time = 0.0

        # ---- 红绿灯检测状态 ----
        self._tl_model_loaded = False  # 模型是否已加载完成

        # ---- Subscribers ----
        rospy.Subscriber("/clicked_point",      PointStamped, self.cb_clicked_point)
        rospy.Subscriber("/usb_cam/image_raw",  Image,        self.cb_image)
        rospy.Subscriber("/scan",               LaserScan,    self.cb_scan)

        # ---- Publishers ----
        # 行人检测结果（NAVIGATING 时发布，供 fusion_person_detect 订阅）
        self.pub_person_det = rospy.Publisher(
            "/yolo/person_detections", Image, queue_size=10)

        self.pub_status   = rospy.Publisher("/traffic_light_status",   String,  queue_size=10)
        self.pub_distance = rospy.Publisher("/traffic_light_distance", Float32, queue_size=10)
        self.pub_nav      = rospy.Publisher("/nav_state",             String,  queue_size=10)
        self.pub_model    = rospy.Publisher("/model_status",          String,  queue_size=10)
        self.pub_cmd_vel  = rospy.Publisher("/cmd_vel",               Twist,   queue_size=10)
        self.pub_waypoints = rospy.Publisher("/waypoint_markers", MarkerArray, queue_size=1)

        # ---- Services ----
        rospy.Service("/start_multi_nav",  Trigger, self.srv_start_nav)
        rospy.Service("/clear_waypoints",  Trigger, self.srv_clear_waypoints)

        # ---- State machine timer (10 Hz) ----
        rospy.Timer(rospy.Duration(0.1), self.cb_state_machine)

        # 预缓存 laser→robot 静态 TF（只查一次，后续 cb_scan 不再逐点查 TF）
        self._cache_laser_transform()

        rospy.loginfo("MultiNav (model-switch) ready. person=%s tl=%s gamma=%.1f",
                      self.person_model_path, self.tl_model_path, self.gamma)
        rospy.loginfo("Layout: 单数WP=红绿灯检查点, 双数WP=普通导航点")

    # --
    #  Model switching (按需加载, 切换时卸载另一模型, 节省GPU显存)
    # --

    def _unload_model(self):
        """卸载当前模型, 释放 GPU/CPU 内存"""
        if self.active_model is not None:
            old_mode = self.active_mode
            del self.active_model
            self.active_model = None
            self.active_mode = None
            gc.collect()
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            if self.pub_model is not None:
                self.pub_model.publish(String("已卸载: {}".format(old_mode)))
            rospy.loginfo("模型已卸载: %s", old_mode)

    def _switch_active_model(self, mode):
        """切换活跃模型。先卸载旧模型释放显存, 再加载新模式 (~3-8s阻塞, 机器人已停车)

        Jetson Xavier 策略：不调 model.half()（会与 YOLO 预处理冲突），
        只用 YOLO 内置 half=True 参数自动处理 FP16 精度。
        warmup OOM → 放弃加载，下帧 cb_image 触发重试。

        ★ 加锁保护：防止 state machine 线程和 cb_image 线程同时切换模型
        导致 active_mode 互相覆盖、反复加载的死循环
        """
        with self._model_switch_lock:
            if self.active_mode == mode and self.active_model is not None:
                return True

            old_mode = self.active_mode
            self._unload_model()

            # 额外清理：GPU 同步 + 清缓存，确保旧模型残留显存被彻底回收
            # 注意：synchronize 可能因 pending OOM 报错，不阻塞切换流程
            if torch is not None:
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            gc.collect()

            path = self.person_model_path if mode == "person" else self.tl_model_path
            msg = "切换: {} -> {}, 加载中...".format(old_mode or "无", mode)
            if self.pub_model is not None:
                self.pub_model.publish(String(msg))
            rospy.loginfo(msg + " (%s)", path)
            t0 = time.time()

            # ---- 加载模型 ----
            self.active_model = YOLO(path)
            self.active_mode = mode
            elapsed = time.time() - t0
            rospy.loginfo("模型加载 %.1fs, 预热中...", elapsed)

            # ---- Warmup: Jetson 上只用 FP32（FP16 可能 SIGSEGV 死进程）----
            warmup_ok = self._try_warmup(use_half=False)
            self._use_half = False

            # ---- 最终结果 ----
            if not warmup_ok:
                rospy.logerr("GPU 显存不足，放弃加载 %s 模型", mode)
                self._unload_model()
                self.active_model = None
                self.active_mode = None
                if self.pub_model is not None:
                    self.pub_model.publish(String("加载失败: {}".format(mode)))
                return False

            done_msg = "{} 已就绪 ({:.1f}s)".format(mode, time.time() - t0)
            if self.pub_model is not None:
                self.pub_model.publish(String(done_msg))
            rospy.loginfo("模型切换完成: %s", done_msg)
            return True

    def _try_warmup(self, use_half=True):
        """小尺寸 warmup（320x320），成功返回 True，OOM 返回 False 且不抛异常"""
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.active_model(dummy, verbose=False, half=use_half)
            rospy.loginfo("Warmup OK (%s)", "FP16" if use_half else "FP32")
            return True
        except Exception as e:
            rospy.logwarn("Warmup OOM: %s", str(e)[:120])
            return False

    def _ensure_model_for_state(self):
        """确保当前 state 所需的模型已加载 (lazy-load)

        ★ 如果有切换正在进行（锁被持有），直接跳过，避免竞态导致反复切换
        """
        # 模型切换进行中 → 跳过（state machine 线程正在加载模型）
        if self._model_switch_lock.locked():
            return

        if self.state in (self.NAVIGATING, self.IDLE):
            # 默认/导航中: 行人检测模型 (yolov8n.pt)
            if self.active_mode != "person":
                if not self._switch_active_model("person"):
                    rospy.logerr_throttle(5, "行人模型加载失败，行人检测暂停")
        elif self.state == self.RED_LIGHT_CHECK:
            # 红绿灯点: 切换为红绿灯模型 (best_new.pt)
            if self.active_mode != "tl":
                if not self._switch_active_model("tl"):
                    rospy.logerr_throttle(5, "红绿灯模型加载失败")

    # --
    #  Pre-compute static laser → robot transform (只查一次, 不再逐帧逐点查 TF)
    # --
    def _cache_laser_transform(self):
        """缓存 laser->robot 的静态 TF, 后续 cb_scan 直接用数学公式做坐标变换."""
        try:
            self.tf_listener.waitForTransform(
                self.robot_frame, self.laser_frame,
                rospy.Time(0), rospy.Duration(2.0))
            (trans, rot) = self.tf_listener.lookupTransform(
                self.robot_frame, self.laser_frame, rospy.Time(0))
            (_, _, yaw) = euler_from_quaternion(rot)
            self._laser_yaw_offset = yaw
            self._laser_tx = trans[0]
            self._laser_ty = trans[1]
            self._laser_tf_cached = True
            rospy.loginfo("LiDAR TF cached: yaw=%.2f° offset=(%.3f, %.3f)",
                          math.degrees(yaw), trans[0], trans[1])
        except Exception as e:
            rospy.logwarn("Cannot cache LiDAR TF: %s (assuming laser==robot aligned)", e)
            self._laser_yaw_offset = 0.0
            self._laser_tx = 0.0
            self._laser_ty = 0.0
            self._laser_tf_cached = True  # 标记已尝试过, 不再重试

    # --
    #  Gamma correction
    # --
    def apply_gamma(self, img):
        if self.gamma_table is not None:
            return cv2.LUT(img, self.gamma_table)
        return img

    # --
    #  Waypoint collection
    # --
    def cb_clicked_point(self, msg):
        if self.state == self.IDLE:
            self.state = self.COLLECTING
            rospy.loginfo("Entering COLLECTING state")

        if self.state == self.COLLECTING:
            x, y = msg.point.x, msg.point.y
            self.waypoints.append((x, y))
            wp_type = "红绿灯检查点" if len(self.waypoints) % 3 == 0 else "普通导航点"
            rospy.loginfo("  [WP %d %s]  (%.2f, %.2f)",
                          len(self.waypoints), wp_type, x, y)
            self._publish_waypoint_markers()

    # --
    #  Service callbacks
    # --
    def srv_start_nav(self, _req):
        if not self.waypoints:
            rospy.logwarn("No waypoints collected.")
            return TriggerResponse(success=False, message="No waypoints")

        # 先加载行人模型，再发导航目标（避免刚起步的3-8s盲走期）
        rospy.loginfo("准备导航，加载行人模型...")
        self._switch_active_model("person")

        self.current_wp_idx = 0
        self.state = self.NAVIGATING
        rospy.loginfo("=== Multi-nav START: %d waypoints ===", len(self.waypoints))
        self._send_waypoint_goal()
        return TriggerResponse(success=True, message="Nav started")

    def srv_clear_waypoints(self, _req):
        self.move_base_client.cancel_all_goals()
        self.waypoints = []
        self.current_wp_idx = 0
        self.state = self.IDLE
        self._hold_robot()
        self._publish_waypoint_markers(clear=True)
        rospy.loginfo("Waypoints cleared. State -> IDLE")
        return TriggerResponse(success=True, message="Cleared")

    # --
    #  Waypoint markers
    # --
    def _publish_waypoint_markers(self, clear=False):
        markers = MarkerArray()
        if not clear:
            for i, (x, y) in enumerate(self.waypoints):
                # 3的倍数WP(3,6,9...)=红绿灯点=红色, 其余=普通=绿色
                is_tl = ((i + 1) % 3 == 0)
                color = (1.0, 0.0, 0.0) if is_tl else (0.0, 1.0, 0.0)
                label = "TL%d" % (i + 1) if is_tl else "WP%d" % (i + 1)

                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = rospy.Time.now()
                m.ns = "waypoints"
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = x
                m.pose.position.y = y
                m.pose.position.z = 0.05
                m.scale.x = 0.15
                m.scale.y = 0.15
                m.scale.z = 0.15
                m.color.r, m.color.g, m.color.b = color
                m.color.a = 0.8
                m.lifetime = rospy.Duration(0)
                markers.markers.append(m)

                t = Marker()
                t.header.frame_id = "map"
                t.header.stamp = rospy.Time.now()
                t.ns = "waypoint_labels"
                t.id = i + 1000
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.pose.position.x = x
                t.pose.position.y = y
                t.pose.position.z = 0.3
                t.scale.z = 0.2
                t.text = label
                t.color.r = t.color.g = t.color.b = 1.0
                t.color.a = 1.0
                t.lifetime = rospy.Duration(0)
                markers.markers.append(t)
        else:
            for ns in ("waypoints", "waypoint_labels"):
                m = Marker()
                m.header.frame_id = "map"
                m.ns = ns
                m.action = Marker.DELETEALL
                markers.markers.append(m)

        if self.pub_waypoints is not None:
            self.pub_waypoints.publish(markers)

    # --
    #  Image callback — 双模式 YOLO 推理
    # --
    def cb_image(self, msg):
        # 只在以下状态处理图像
        if self.state not in (self.NAVIGATING, self.RED_LIGHT_CHECK, self.IDLE):
            return

        # P0: 帧率节流 — 丢弃来不及处理的帧，避免延迟累积
        now = rospy.Time.now().to_sec()
        if now - self._last_infer_time < self._min_infer_interval:
            return
        self._last_infer_time = now

        # 按需加载模型（首次或切换后，含 OOM 损坏自动重载）
        if self._model_corrupted:
            rospy.logwarn("检测到 GPU OOM，卸载损坏模型...")
            self._unload_model()
            self._model_corrupted = False
        self._ensure_model_for_state()

        if self.active_model is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn("cv_bridge error: %s", e)
            return

        processed = self.apply_gamma(frame)

        # 推理异常保护：捕获 OOM / CUBLAS / CUDA 致命错误 + None 模型
        if self.active_model is None:
            return
        try:
            results = self.active_model(processed, verbose=False, half=self._use_half, imgsz=416)
        except RuntimeError as e:
            err_str = str(e)
            if ("out of memory" in err_str.lower()
                    or "CUBLAS_STATUS" in err_str
                    or "CUDNN_STATUS" in err_str
                    or "CUDA error" in err_str):
                rospy.logwarn("GPU fatal error → 标记模型损坏，下帧自动重载: %s", err_str[:120])
                self._model_corrupted = True
            else:
                rospy.logwarn_throttle(5, "GPU inference failed: %s", err_str[:120])
            return

        # ---- 按当前模式分发 ----
        if self.active_mode == "person":
            self._process_person_mode(results, msg.header)
        elif self.active_mode == "tl":
            self._process_tl_mode(results, msg.header)

    # ----------------------------------------------------------------
    #  PERSON mode: 发布 /yolo/person_detections (与 yolo_detect.py 兼容)
    # ----------------------------------------------------------------
    def _process_person_mode(self, results, header):
        """★ P1 优化: 向量化 numpy 编码，消除 Python dict/list 循环"""
        if not results or len(results) == 0:
            det_msg = self.bridge.cv2_to_imgmsg(
                np.zeros((1, 1), dtype=np.float32), encoding='32FC1')
            det_msg.header = header
            if self.pub_person_det is not None:
                self.pub_person_det.publish(det_msg)
            return

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            det_msg = self.bridge.cv2_to_imgmsg(
                np.zeros((1, 1), dtype=np.float32), encoding='32FC1')
            det_msg.header = header
            if self.pub_person_det is not None:
                self.pub_person_det.publish(det_msg)
            return

        # 一次性取全部数据 (N, 6) = [x1,y1,x2,y2,conf,cls]
        data = boxes.data.cpu().float().numpy()  # torch → numpy, 兼容 FP16/FP32
        mask = (data[:, 5] == self.PERSON_CLASS_ID) & (data[:, 4] >= self.person_conf_thresh)
        filtered = data[mask]

        if len(filtered) == 0:
            det_msg = self.bridge.cv2_to_imgmsg(
                np.zeros((1, 1), dtype=np.float32), encoding='32FC1')
            det_msg.header = header
            if self.pub_person_det is not None:
                self.pub_person_det.publish(det_msg)
            return

        # 向量化编码: [cx, cy, w, h, conf]
        x1, y1, x2, y2, conf = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3], filtered[:, 4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        det_arr = np.column_stack([cx, cy, w, h, conf]).astype(np.float32).reshape(-1, 1)

        det_msg = self.bridge.cv2_to_imgmsg(det_arr, encoding='32FC1')
        det_msg.header = header
        if self.pub_person_det is not None:
            self.pub_person_det.publish(det_msg)

    # ----------------------------------------------------------------
    #  TL mode: 检测红绿灯颜色 + 估计角度
    # ----------------------------------------------------------------
    def _process_tl_mode(self, results, header):
        """红绿灯检测 — 输出 detected_class, target_angle"""
        best_target = None
        min_dist = float("inf")

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls > 2:
                continue
            conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 1.0
            if conf < self.tl_conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            d = math.sqrt((u - self.cx) ** 2 + (v - self.cy) ** 2)
            if d < min_dist:
                min_dist = d
                best_target = {
                    "u": u, "v": v, "cls": cls, "conf": conf,
                }

        if best_target:
            u, v, cls, conf = (best_target[k] for k in ("u", "v", "cls", "conf"))
            self.detected_class = self.TL_CLASS_NAMES.get(cls, None)

            # 像素 → 相机归一化 → ROS坐标
            x_norm = (u - self.cx) / self.fx
            y_norm = (v - self.cy) / self.fy
            p_cam = self.R_CV_TO_ROS @ np.array([x_norm, y_norm, 1.0])

            pt = PointStamped()
            pt.header.frame_id = self.camera_frame
            pt.header.stamp = header.stamp
            pt.point.x = p_cam[0]
            pt.point.y = p_cam[1]
            pt.point.z = p_cam[2]

            try:
                # 用图像消息实际时间戳做 TF 查询，避免取"最新"变换导致偏移
                transform = self.tf_buffer.lookup_transform(
                    self.robot_frame, self.camera_frame,
                    header.stamp, rospy.Duration(0.1))
                pt_base = do_transform_point(pt, transform)
                self.target_angle = math.atan2(pt_base.point.y, pt_base.point.x)
            except (tf.LookupException, tf.ConnectivityException,
                    tf.ExtrapolationException, tf2_ros.LookupException,
                    tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.target_angle = None
                self.detected_class = None
        else:
            self.detected_class = None
            self.target_angle = None

        if self.pub_status is not None:
            self.pub_status.publish(String(self.detected_class or "none"))

    # --
    #  LiDAR scan callback (TL mode only — 估红绿灯距离)
    # --
    def cb_scan(self, msg):
        """★ P0 优化: 使用缓存的静态 TF 做数学坐标变换, 消除逐点 waitForTransform+transformPoint"""
        # 快照 target_angle 防止 image 回调竞态置 None
        target_angle = self.target_angle
        if self.active_mode != "tl" or target_angle is None:
            return

        if not self._laser_tf_cached:
            return

        cos_yaw = math.cos(self._laser_yaw_offset)
        sin_yaw = math.sin(self._laser_yaw_offset)
        tx, ty = self._laser_tx, self._laser_ty

        min_dist = float("inf")
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        r_min = msg.range_min
        r_max = msg.range_max

        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r) or r < r_min or r > r_max:
                continue

            # 激光点坐标 (laser_frame)
            angle_l = angle_min + i * angle_inc
            lx = r * math.cos(angle_l)
            ly = r * math.sin(angle_l)

            # 用缓存 TF 直接做数学变换 (P_robot = T + R * P_laser)
            rx = tx + lx * cos_yaw - ly * sin_yaw
            ry = ty + lx * sin_yaw + ly * cos_yaw

            # 该激光点在 robot_frame 中的方位角
            a_robot = math.atan2(ry, rx)

            if abs(a_robot - target_angle) < self.angle_tolerance:
                if r < min_dist:
                    min_dist = r

        self._dist_buffer.append(min_dist)
        valid = [d for d in self._dist_buffer if d != float("inf")]
        self._dist_filtered = sum(valid) / len(valid) if valid else float("inf")
        self.light_distance = self._dist_filtered
        if self.pub_distance is not None:
            self.pub_distance.publish(Float32(self.light_distance))

    # --
    #  State machine (10 Hz)
    # --
    def cb_state_machine(self, _event):
        if self.pub_nav is not None:
            self.pub_nav.publish(String(self.state))

        handlers = {
            self.IDLE:             self._do_idle,
            self.COLLECTING:       self._do_collecting,
            self.NAVIGATING:       self._do_navigating,
            self.RED_LIGHT_CHECK:  self._do_red_light_check,
            self.DONE:             lambda: None,
        }
        handlers.get(self.state, lambda: None)()

    def _do_idle(self):
        pass

    def _do_collecting(self):
        pass

    # ----------------------------------------------------------------
    #  NAVIGATING: 走向下一个WP, yolov8n.pt 在跑 (行人避障由 costmap 自动处理)
    # ----------------------------------------------------------------
    def _do_navigating(self):
        status = self.move_base_client.get_state()

        if status == actionlib.GoalStatus.SUCCEEDED:
            wp_num = self.current_wp_idx + 1
            rospy.loginfo("Waypoint %d/%d reached.", wp_num, len(self.waypoints))

            # 3的倍数WP (3,6,9...) = 红绿灯检查点
            if wp_num % 3 == 0:
                rospy.loginfo(">>> WP%d 红绿灯检查点, 停车 + 切换模型 → tl", wp_num)
                self._hold_robot()
                tl_ok = self._switch_active_model("tl")
                if not tl_ok:
                    rospy.logerr(">>> 红绿灯模型加载失败！仍进入旋转扫描（将走超时放行）")
                self.detected_class = None
                self.target_angle = None
                self.light_distance = float("inf")
                self._dist_buffer.clear()
                self._green_count = 0
                self._red_seen_count = 0
                self._red_wait_start = time.time()
                self.state = self.RED_LIGHT_CHECK
                return

            # 非3倍数WP = 普通点, 直接下一个
            self._advance_to_next_wp()

        elif status in (actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED):
            now = time.time()
            if now - self._last_retry_time >= self._retry_cooldown:
                rospy.logwarn("Goal %d failed (code=%d), retrying ...",
                              self.current_wp_idx + 1, status)
                self._last_retry_time = now
                self._send_waypoint_goal()

        elif status not in (actionlib.GoalStatus.ACTIVE, actionlib.GoalStatus.PENDING):
            self._send_waypoint_goal()

    # ----------------------------------------------------------------
    #  RED_LIGHT_CHECK: 停车 → 切模型 → 检测 → 等绿灯/超时放行
    # ----------------------------------------------------------------
    def _do_red_light_check(self):
        # ---- 等待模型加载 ----
        if not self._tl_model_loaded:
            if self.active_mode == "tl" and self.active_model is not None:
                self._tl_model_loaded = True
                rospy.loginfo(">>> 红绿灯模型就绪，开始检测")
            else:
                self._hold_robot()
                return

        # ---- 绿灯检测与确认 ----
        if self.detected_class == "green":
            self._green_count += 1
            rospy.loginfo(">>> 检测到: green (%d/%d)",
                          self._green_count, self.green_confirm_n)
            if self._green_count >= self.green_confirm_n:
                rospy.loginfo(">>> GREEN confirmed! 切回person模型 → 继续")
                self._hold_robot()
                self._green_count = 0
                self._reset_tl()
                # 先切状态再切模型：防止 cb_image 在窗口期误切回 tl 模型
                self.state = self.NAVIGATING
                self._switch_active_model("person")
                self._advance_to_next_wp()
                return
        else:
            self._green_count = 0
            if self.detected_class == "red":
                self._red_seen_count += 1
                rospy.loginfo(">>> 检测到: red (%d frames)",
                              self._red_seen_count)

        # ---- 超时放行 ----
        elapsed = time.time() - self._red_wait_start
        if elapsed > self.red_wait_timeout:
            rospy.logwarn(">>> 红绿灯等待超时 (%.1fs), 强制通过", elapsed)
            self._green_count = 0
            self._reset_tl()
            self._switch_active_model("person")
            self._advance_to_next_wp()

    def _reset_tl(self):
        """重置红绿灯检测状态"""
        self._tl_model_loaded = False

    # ----------------------------------------------------------------
    #  Navigation helper
    # ----------------------------------------------------------------
    def _advance_to_next_wp(self):
        """前进到下一个 waypoint"""
        self.current_wp_idx += 1
        if self.current_wp_idx >= len(self.waypoints):
            self.state = self.DONE
            rospy.loginfo("=== All waypoints DONE! ===")
            return
        self.state = self.NAVIGATING
        self._send_waypoint_goal()

    def _send_waypoint_goal(self):
        if self.current_wp_idx >= len(self.waypoints):
            return
        x, y = self.waypoints[self.current_wp_idx]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        # 红绿灯点(3,6,9...): 朝向设为 "上一个点 → 当前点" 的指向，让相机正对灯
        # 普通导航点: 朝向设为 "当前点 → 下一个点" 的方向，到站后面朝下一个目标
        wp_num = self.current_wp_idx + 1
        if wp_num % 3 == 0 and self.current_wp_idx > 0:
            # 红绿灯点: 面向上一个WP方向（相机正对来向的灯）
            px, py = self.waypoints[self.current_wp_idx - 1]
            yaw = math.atan2(y - py, x - px)
            q = quaternion_from_euler(0, 0, yaw)
            goal.target_pose.pose.orientation.x = q[0]
            goal.target_pose.pose.orientation.y = q[1]
            goal.target_pose.pose.orientation.z = q[2]
            goal.target_pose.pose.orientation.w = q[3]
            rospy.loginfo("-> WP %d/%d [TL]  (%.2f, %.2f) 朝向=%.0f° (←上一WP)",
                          wp_num, len(self.waypoints), x, y, math.degrees(yaw))
        elif self.current_wp_idx + 1 < len(self.waypoints):
            # 普通点 (还有下一个WP): 朝向指向下一个点
            nx, ny = self.waypoints[self.current_wp_idx + 1]
            yaw = math.atan2(ny - y, nx - x)
            q = quaternion_from_euler(0, 0, yaw)
            goal.target_pose.pose.orientation.x = q[0]
            goal.target_pose.pose.orientation.y = q[1]
            goal.target_pose.pose.orientation.z = q[2]
            goal.target_pose.pose.orientation.w = q[3]
            rospy.loginfo("-> WP %d/%d  (%.2f, %.2f) 朝向=%.0f° (→下一WP)",
                          wp_num, len(self.waypoints), x, y, math.degrees(yaw))
        else:
            # 最后一个WP（非红绿灯点）: 保持默认朝向
            goal.target_pose.pose.orientation.w = 1.0
            rospy.loginfo("-> WP %d/%d  (%.2f, %.2f) [终点]",
                          wp_num, len(self.waypoints), x, y)

        self.move_base_client.send_goal(goal)

    def _hold_robot(self):
        if self.pub_cmd_vel is None:
            return
        cmd = Twist()
        self.pub_cmd_vel.publish(cmd)

    # --
    #  Shutdown
    # --
    def shutdown(self):
        self._hold_robot()
        self._unload_model()
        rospy.loginfo("MultiNav node shutdown.")


# -- Entry point --
if __name__ == "__main__":
    try:
        # Jetson 环境配置（建议在 launch 文件中通过 env 参数设置）
        os.environ.setdefault("LD_PRELOAD",
                              "/usr/lib/aarch64-linux-gnu/libgomp.so.1")
    except Exception:
        pass

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "max_split_size_mb:64,garbage_collection_threshold:0.6")

    try:
        node = MultiNavTrafficLightNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
