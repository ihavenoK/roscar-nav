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
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### Launch

```bash
# Full stack: navigation + radar fusion + model switching
roslaunch multi_nav_traffic nav.launch

# Radar + fusion only (debug)
roslaunch ars408_ros radar_nav.launch
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
