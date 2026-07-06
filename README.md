# ROS-Car: Multi-Point Navigation with Radar-Vision Fusion

ROS1 Noetic 智能小车 — 多点导航 + 红绿灯识别 + 毫米波雷达行人避障。

## Features

- **Multi-point navigation** with dynamic waypoint handling (traffic-light checkpoints every 3rd WP)
- **Dynamic model switching** — single YOLO instance saves ~50% GPU memory on Jetson
- **mmWave radar + YOLO fusion** — ARS408 radar points projected via TF into camera frame
- **Pedestrian obstacle point cloud** → costmap obstacle_layer for autonomous avoidance
- **Traffic light detection** (red/green/yellow) with scan-to-find behavior at intersections

## Architecture

```
/usb_cam/image_raw
    ↓
multi_nav_traffic_light_node (single YOLO, model switches by state)
    ├── NAVIGATING:      person bbox → /yolo/person_detections
    │                        ↓
    │                    fusion_person_detect (radar + YOLO → PointCloud2)
    │                        ↓
    │                    costmap obstacle_layer → move_base avoidance
    │
    └── RED_LIGHT_CHECK: red/green/yellow → wait for green → switch back
```

## Hardware

| Component | Model |
|-----------|-------|
| Compute | NVIDIA Jetson Nano 4GB |
| Radar | ARS408-21 (mmWave, CAN bus) |
| LiDAR | RPLIDAR A1 |
| Camera | USB webcam |
| Chassis | 3-wheel differential drive |

## Quick Start

### Prerequisites

- Ubuntu 20.04 + ROS Noetic
- Python 3.8+ with PyTorch, Ultralytics YOLOv8, OpenCV
- CAN interface configured (`can0` at 500kbps for ARS408)

### Build

```bash
cd ~/catkin_roscar
catkin_make
source devel/setup.bash
```

### One-Click Shell Scripts

项目提供了 4 个一键启动脚本，包含 CAN 配置、roscore 检查、工作空间加载等前置步骤。

#### 1. `start_nav.sh` — 多点导航 + 雷达避障 + 红绿灯（真车运行）

```bash
# 加载默认地图 ~/map/my_map.yaml
bash start_nav.sh

# 加载指定地图
bash start_nav.sh roscar_map
```

| 步骤 | 内容 |
|------|------|
| CAN 配置 | 检查/启动 `can0` (500kbps)，无法启动则退出 |
| roscore | 检测已在运行的 roscore，否则自动启动 |
| 工作空间 | `source ~/catkin_roscar/devel/setup.bash`，未编译则退出 |
| 地图检查 | 验证 `~/map/{name}.yaml` 存在，否则列出可用地图并退出 |
| 启动 | `roslaunch multi_nav_traffic nav.launch map_file:={path}` |

启动内容：STM32 底盘驱动 + TF + RPLidar + ARS408 雷达 + USB 摄像头 + map_server + AMCL + move_base + YOLO 行人检测/雷达融合 + 红绿灯识别 + 动态模型切换。

#### 2. `start_radar_nav.sh` — 雷达导航（不含红绿灯/多点规划）

```bash
bash start_radar_nav.sh
```

启动 `radar_nav.launch`：STM32 底盘 + RPLidar + ARS408 + 摄像头 + AMCL + move_base + YOLO 行人检测 + 雷达融合。**不启动**多点导航和红绿灯节点。

#### 3. `start_mapping.sh` — 键盘遥控建图

```bash
# 默认参数：线速度 0.15 m/s，角速度 0.6 rad/s，保存到 ~/map/my_map
bash start_mapping.sh

# 自定义速度 + 地图名
bash start_mapping.sh 0.10 0.5 ~/map manual_map

# Jetson 无显示器时禁用 RViz
USE_RVIZ=false bash start_mapping.sh
```

| 步骤 | 内容 |
|------|------|
| roscore | 自动检查并启动 |
| 工作空间 | `source ~/catkin_roscar/devel/setup.bash` |
| gmapping | 后台启动 `gmapping.launch`（底盘 + TF + 雷达 + 摄像头 + slam_gmapping + 可选 RViz） |
| 键盘遥控 | 前台运行 `manual_mapping.py` |

键盘控制：`W/S` 前进后退，`A/D` 左转右转，`M` 保存地图，`Q` 退出。

#### 4. `start_traffic_light_debug.sh` — 红绿灯识别 + 激光测距调试

```bash
bash start_traffic_light_debug.sh
```

| 步骤 | 内容 |
|------|------|
| 硬件检查 | 检查 RPLidar (`/dev/ttyUSB0`) 和 USB 摄像头 (`/dev/video*`) 连接状态 |
| roscore | 自动检查并启动 |
| 工作空间 | `source ~/catkin_roscar/devel/setup.bash` |
| 启动 | `traffic_light_debug.launch`：USB 摄像头 + RPLidar 测距 + YOLO 红绿灯识别 |

### roslaunch 直接启动

```bash
# 完整导航栈（nav.launch 内部 include radar_nav → navigation → start_roscar）
roslaunch multi_nav_traffic nav.launch

# 雷达 + 融合可视化调试
roslaunch ars408_ros fusion_debug.launch

# 仅雷达导航（含 AMCL + move_base，不含多点规划）
roslaunch ars408_ros radar_nav.launch

# 基础底盘 + 传感器
roslaunch start_roscar start_roscar.launch

# 电磁巡线
rosrun ele_line_follower ele_line_follower.py
```

### Model Setup

See [model/README.md](src/multi_nav_traffic/model/README.md) for YOLO weight setup.

## Package Overview

| Package | Description |
|---------|-------------|
| `ars408_ros` | ARS408 radar driver + YOLO fusion + person obstacle cloud |
| `multi_nav_traffic` | Multi-point nav with dynamic model switching |
| `driver` | Motor control (odom/velocity inverters) |
| `ele_line_follower` | Magnetic line following + manual mapping + teleop |
| `start_roscar` | Launch integration (AMCL, move_base, robot model) |
| `roscar_slam` | SLAM with gmapping |

## Key Nodes

| Node | Topic In → Out | Purpose |
|------|----------------|---------|
| `ars408_node` | CAN → `/radar/pointcloud` | ARS408 radar parser |
| `fusion_node` | `/radar/pointcloud` + `/yolo/person_detections` → `/fusion/printpoint` | Radar-YOLO fusion visualization |
| `fusion_person_detect` | `/radar/pointcloud` + `/yolo/person_detections` → `/person_obstacle_cloud` | Person obstacle for costmap |
| `multi_nav_traffic_light_node` | `/usb_cam/image_raw` → `/yolo/person_detections` or red/green state | Multi-WP nav + model switching |
| `pedestrian_to_scan` | MarketArray → `/pedestrian_scan` | Person → LaserScan conversion |

## Author

蔡博涵 (Cai Bohan) · 广东工业大学 电子信息工程

## License

This project is for educational use (course project).
