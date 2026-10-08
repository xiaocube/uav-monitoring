#!/bin/bash
# SkyGuard 无人机检测系统 - 一键启动脚本

APP_NAME="SkyGuard 无人机监测系统"
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "╔══════════════════════════════════════════════╗"
echo "║                                              ║"
echo "║     SkyGuard 无人机监测系统 Web Demo         ║"
echo "║                                              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"

# 确保 node/npm 在 PATH 中（常见安装位置）
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# 定位 Python（优先使用虚拟环境，但不修改 PATH 以免影响 node）
PYTHON_BIN="python3"
if [ -f "../.venv/bin/python" ]; then
    echo "📦 使用 Python 虚拟环境..."
    PYTHON_BIN="$(cd .. && pwd)/.venv/bin/python"
else
    echo "⚠️  未找到虚拟环境，使用系统 Python"
fi

# 检查 Node.js
if ! command -v npm &> /dev/null; then
    echo "❌ 未找到 Node.js，请先安装 Node.js"
    echo "   下载地址: https://nodejs.org/"
    exit 1
fi
echo "   ✅ Node.js: $(node --version)"

# 安装前端依赖（如果需要）
if [ ! -d "node_modules" ]; then
    echo "📦 安装前端依赖..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ 前端依赖安装失败"
        exit 1
    fi
fi

# 确保示例图片存在
if [ ! -d "public/samples" ]; then
    mkdir -p public/samples
    if [ -d "../data/samples/demo_results" ]; then
        cp -f ../data/samples/demo_results/*.jpg public/samples/ 2>/dev/null
        echo "   ✅ 已复制示例图片"
    fi
fi

# 安装后端依赖
echo "📦 检查后端依赖..."
"$PYTHON_BIN" -c "import flask" 2>/dev/null || "$PYTHON_BIN" -m pip install flask flask-cors flask-socketio opencv-python -q

# 检查模型文件
echo ""
echo "🔍 检查模型文件..."
MODEL_CONFIG="../config/models.json"
if [ -f "$MODEL_CONFIG" ]; then
    "$PYTHON_BIN" -c "
import json, sys
with open('$MODEL_CONFIG') as f:
    cfg = json.load(f)
import os
for name, info in cfg['models'].items():
    path = os.path.join('..', info['path'])
    status = '✅' if os.path.exists(path) else '❌'
    print(f'  {status} {name}: {info[\"name\"]}')
"
fi

echo ""
echo "🚀 启动后端服务器 (端口 5001)..."
cd server
"$PYTHON_BIN" app.py > /tmp/skyguard_backend.log 2>&1 &
BACKEND_PID=$!
cd ..

sleep 3

# 检查后端是否启动成功
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "   ✅ 后端服务器启动成功 (PID: $BACKEND_PID)"
else
    echo "   ⚠️  后端服务器启动失败，日志："
    tail -5 /tmp/skyguard_backend.log
    echo "   继续使用前端演示模式"
fi

echo ""
echo "🎨 启动前端开发服务器 (端口 5173)..."
npm run dev > /tmp/skyguard_frontend.log 2>&1 &
FRONTEND_PID=$!

sleep 3

# 检查前端是否启动成功
if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "   ✅ 前端服务器启动成功 (PID: $FRONTEND_PID)"
else
    echo "   ❌ 前端服务器启动失败"
    tail -10 /tmp/skyguard_frontend.log
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════"
echo ""
echo "  ✅ 系统启动完成！"
echo ""
echo "  🌐 前端地址: http://localhost:5173"
echo "  🔧 后端地址: http://localhost:5001"
echo ""
echo "  💡 使用提示："
echo "     1. 点击右上角 \"选择模型\" 切换模型"
echo "     2. 支持摄像头实时检测 / 图片上传测试"
echo "     3. 可选择 WBF 集成方案或单模型"
echo ""
echo "  🛑 按 Ctrl+C 停止所有服务"
echo ""
echo "═══════════════════════════════════════════════"
echo ""

# 自动打开浏览器
sleep 1
open "http://localhost:5173"

# 等待用户中断
trap cleanup INT

cleanup() {
    echo ""
    echo ""
    echo "🛑 正在停止服务..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait 2>/dev/null
    echo "✅ 所有服务已停止"
    exit 0
}

wait
