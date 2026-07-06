#!/bin/bash
# ============================================================
# 雷达行人检测 + 多点导航 + 红绿灯识别 一键启动脚本
# 用法: bash start_radar_nav.sh
# ============================================================

# ARM64 修复：PyTorch 导入时的 TLS 内存分配错误（必须在所有进程前设置）
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

echo "================================================"
echo "  雷达+导航+红绿灯识别 一键启动"
echo "================================================"

# 第1步：配置CAN接口（与 start_radar_debug.sh 完全一致）
echo ""
echo "===== [1/4] 配置CAN接口 ====="
echo "123456" | sudo -S modprobe can 2>/dev/null
echo "123456" | sudo -S modprobe can_raw 2>/dev/null
echo "123456" | sudo -S modprobe mttcan 2>/dev/null
echo "123456" | sudo -S ip link set can0 up type can bitrate 500000 2>&1
if ip link show can0 &>/dev/null; then
    echo "✅ CAN接口已配置 (can0, 500kbps)"
else
    echo "❌ CAN接口配置失败，can0 不存在！请检查雷达硬件连接"
    exit 1
fi

# 第2步：检查并启动 roscore
echo ""
echo "===== [2/4] 检查 roscore ====="
if rostopic list &>/dev/null 2>&1; then
    echo "✅ roscore 已在运行"
else
    echo "正在启动 roscore..."
    roscore &
    sleep 3
    echo "✅ roscore 已启动"
fi

# 第3步：加载工作空间
echo ""
echo "===== [3/4] 加载工作空间 ====="
source ~/catkin_roscar/devel/setup.bash
echo "✅ 工作空间已加载"

# 第4步：启动 launch
echo ""
echo "===== [4/4] 启动 radar_nav.launch ====="
echo "> STM32底盘驱动 + TF坐标系"
echo "> RPLidar A3 激光雷达"
echo "> ARS408 毫米波雷达 (CAN)"
echo "> USB 摄像头"
echo "> map_server + AMCL 定位"
echo "> move_base 路径规划"
echo "> YOLO 行人检测 + 雷达融合"
echo "> 红绿灯识别 + 测距"
echo ""
roslaunch ars408_ros radar_nav.launch
