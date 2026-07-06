#!/bin/bash
# ============================================================
# 红绿灯识别+测距 一键调试脚本
# 用法: bash start_traffic_light_debug.sh
# ============================================================

set -e

# ARM64 修复：PyTorch 导入时的 TLS 内存分配错误（必须在所有进程前设置）
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

echo "================================================"
echo "  红绿灯识别 + 激光雷达测距 一键调试"
echo "================================================"

# 第0步：硬件连接检查
echo ""
echo "===== [0/3] 硬件连接检查 ====="

# 检查 RPLidar
if [ -e /dev/ttyUSB0 ]; then
    echo "✅ RPLidar 已连接 (/dev/ttyUSB0)"
else
    echo "❌ 未检测到 RPLidar (/dev/ttyUSB0)！"
    echo "   请检查 USB 线是否插入，或修改 rplidar.launch 中的 serial_port"
    echo ""
fi

# 检查 USB 摄像头
if ls /dev/video* 1>/dev/null 2>&1; then
    echo "✅ USB 摄像头已连接 ($(ls /dev/video* 2>/dev/null | tr '\n' ' '))"
else
    echo "❌ 未检测到 USB 摄像头 (/dev/video*)！请检查连接"
    echo ""
fi

# 第1步：检查并启动 roscore
echo ""
echo "===== [1/3] 检查 roscore ====="
if rostopic list &>/dev/null 2>&1; then
    echo "✅ roscore 已在运行"
else
    echo "正在启动 roscore..."
    roscore &
    sleep 3
    echo "✅ roscore 已启动"
fi

# 第2步：source 工作空间
echo ""
echo "===== [2/3] 加载工作空间 ====="
source ~/catkin_roscar/devel/setup.bash
echo "✅ 工作空间已加载"

# 第3步：启动 launch
echo ""
echo "===== [3/3] 启动红绿灯识别+测距 ====="
echo "> 启动 USB 摄像头"
echo "> 启动 RPLidar 激光雷达 (测距)"
echo "> 启动 YOLO 红绿灯识别节点"
echo "> 检测窗口将弹出（如有显示器）"
echo ""
roslaunch multi_nav_traffic traffic_light_debug.launch
