#!/bin/bash
# ============================================================
# 多点导航 + 雷达行人避障 + 红绿灯识别 一键启动脚本
# 用法: bash start_nav.sh
# ============================================================

set -e

# ARM64 修复：PyTorch 导入时的 TLS 内存分配错误（必须在所有进程前设置）
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

echo "================================================"
echo "  多点导航 + 雷达行人避障 + 红绿灯识别"
echo "================================================"

# 第0步：清理残留节点（slam_gmapping 等会和 AMCL / map_server 冲突）
echo ""
echo "===== [0/5] 清理残留节点 ====="
rosnode kill /slam_gmapping 2>/dev/null && echo "已清理 /slam_gmapping" || true

# 第1步：配置CAN接口
echo ""
echo "===== [1/4] 配置CAN接口 ====="
echo "123456" | sudo -S modprobe can 2>/dev/null
echo "123456" | sudo -S modprobe can_raw 2>/dev/null
echo "123456" | sudo -S modprobe mttcan 2>/dev/null

# 检查 can0 是否已存在并 UP
if ip link show can0 &>/dev/null; then
    CAN_STATE=$(ip link show can0 2>/dev/null | grep -o 'state \w*')
    if echo "$CAN_STATE" | grep -q 'UP'; then
        echo "✅ CAN接口已UP (can0)，跳过配置"
    else
        echo "正在配置 can0 (500kbps)..."
        echo "123456" | sudo -S ip link set can0 type can bitrate 500000 2>/dev/null || true
        echo "123456" | sudo -S ip link set can0 up 2>/dev/null || true
        echo "✅ CAN接口已启动 (can0, 500kbps)"
    fi
else
    echo "❌ CAN接口不存在 (can0)，请检查雷达硬件连接"
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

# 第4步：地图选择
# 默认加载 my_map，可通过 $1 参数指定其他地图名
# 用法: bash start_nav.sh                    → 加载 ~/map/my_map
#       bash start_nav.sh manual_map          → 加载 ~/map/manual_map.yaml
#       bash start_nav.sh roscar_map          → 加载 ~/map/roscar_map.yaml
MAP_NAME="${1:-my_map}"
MAP_FILE="$HOME/map/${MAP_NAME}.yaml"

if [ ! -f "$MAP_FILE" ]; then
    echo "❌ 地图文件不存在: $MAP_FILE"
    echo "   请先建图并保存，或检查 ~/map/ 目录"
    ls ~/map/*.yaml 2>/dev/null || echo "   (无 .yaml 文件)"
    exit 1
fi

echo ""
echo "===== [4/4] 启动 multi_nav_traffic nav.launch ====="
echo "> 加载地图: $MAP_FILE"
echo "> STM32底层驱动 + TF坐标系"
echo "> RPLidar A3 激光雷达"
echo "> ARS408 毫米波雷达 (CAN)"
echo "> USB 摄像头"
echo "> map_server + AMCL 定位"
echo "> move_base 路径规划"
echo "> YOLO 行人检测 + 雷达融合避障"
echo "> 多点导航 + 红绿灯识别 + 动态模型切换"
echo ""
roslaunch multi_nav_traffic nav.launch map_file:="$MAP_FILE"
