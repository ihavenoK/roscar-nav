#!/bin/bash
# ============================================================
# 键盘遥控 + gmapping 建图 一键启动脚本
# 用法: bash ./start_mapping.sh
# 或:   ./start_mapping.sh
# ============================================================

set -e

# ========== ARM64 兼容修复 ==========
# 无 GPU 的 ARM64 平台 RViz 需要软件渲染
# 注意: 不设置 LD_PRELOAD=libgomp, 建图无需 PyTorch,
#       强制 preload 会导致 car_driver 等 C++ 节点 SIGABRT 崩溃
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_QUICK_BACKEND=software
# 若仍有 OpenGL 问题, 传 USE_RVIZ=false 禁用 RViz
# =====================================

# ========== 可调参数 ==========
LINEAR_SPEED="${1:-0.15}"        # 线速度 m/s (建图建议慢速 0.10~0.20)
ANGULAR_SPEED="${2:-0.6}"       # 角速度 rad/s
MAP_DIR="${3:-$HOME/map}"       # 地图保存目录
MAP_NAME="${4:-my_map}"         # 地图文件名
USE_RVIZ="${USE_RVIZ:-true}"    # 是否启动 RViz (无 GPU 建议 false)
# ===============================

echo "================================================"
echo "    键盘遥控建图一键启动"
echo "    gmapping SLAM + 手动遥控 + 一键保存地图"
echo "================================================"
echo "  速度参数: lin=${LINEAR_SPEED} m/s  ang=${ANGULAR_SPEED} rad/s"
echo "  地图保存: ${MAP_DIR}/${MAP_NAME}.yaml / .pgm"
echo ""

# ==== 第1步：检查 roscore ====
echo "===== [1/4] 检查 roscore ====="
if rostopic list &>/dev/null 2>&1; then
    echo "✅ roscore 已在运行"
else
    echo "正在启动 roscore..."
    roscore &
    sleep 3
    echo "✅ roscore 已启动"
fi

# ==== 第2步：加载工作空间 ====
echo ""
echo "===== [2/4] 加载工作空间 ====="
WS_DIR="$HOME/catkin_roscar"
source "$WS_DIR/devel/setup.bash" 2>/dev/null || {
    echo "❌ 工作空间未编译, 请先执行 catkin_make"
    exit 1
}
echo "✅ 工作空间已加载: $WS_DIR"

# ==== 第3步：启动 gmapping 建图环境 (后台) ====
echo ""
echo "===== [3/4] 启动 gmapping 建图环境 ====="
echo "> STM32 底盘驱动"
echo "> TF 坐标系 + robot_pose_ekf"
echo "> RPLidar A3 激光雷达"
echo "> USB 摄像头"
echo "> slam_gmapping 建图节点"
echo "> RViz 可视化"
echo ""
echo "启动中... 关闭视觉循迹, 使用键盘遥控代替"

roslaunch roscar_slam gmapping.launch \
    use_visual_follower:=false \
    use_rviz:=${USE_RVIZ} &
GMFG_PID=$!
sleep 5

# 等待 gmapping 节点就绪
for i in $(seq 1 15); do
    if rostopic list 2>/dev/null | grep -q "/cmd_vel"; then
        echo "✅ gmapping 建图环境已就绪"
        break
    fi
    sleep 1
done

# ==== 第4步：启动键盘遥控 (前台) ====
echo ""
echo "===== [4/4] 启动键盘遥控 ====="
echo ""
echo "  ┌────────────────────────────────────────────────┐"
echo "  │  控制键:                                       │"
echo "  │    W/↑ 前进        S/↓ 后退                   │"
echo "  │    A/← 左转        D/→ 右转                   │"
echo "  │    X   停止        R 加速    E 减速            │"
echo "  │    M   保存地图   (保存到 ${MAP_DIR})          │"
echo "  │    Q   退出并停车                              │"
echo "  │                                                │"
echo "  │  ⚠ 建图建议:                                   │"
echo "  │    1. 慢速行驶 (默认 0.15 m/s)                  │"
echo "  │    2. 转弯时多停留让激光回环收敛               │"
echo "  │    3. 走完一圈回起点后按 M 保存地图            │"
echo "  │    4. 建图完成后地图在 ${MAP_DIR} 目录         │"
echo "  └────────────────────────────────────────────────┘"
echo ""

rosrun ele_line_follower manual_mapping.py \
    _linear_speed:=${LINEAR_SPEED} \
    _angular_speed:=${ANGULAR_SPEED} \
    _map_dir:=${MAP_DIR} \
    _map_name:=${MAP_NAME}

# 用户退出键盘遥控后, 清理后台 gmapping
echo ""
echo "===== 正在关闭建图环境 ====="
kill $GMFG_PID 2>/dev/null || true
sleep 1
pkill -f slam_gmapping 2>/dev/null || true
echo "已退出, 可查看地图:"
echo "  ls ${MAP_DIR}/"
echo "================================================"
