#!/bin/bash
# SkyGuard API 启动器
# 用法: 双击运行，或终端执行 ./start_skyguard.sh

# 切换到脚本所在目录（项目根）
cd "$(dirname "$0")"

# 检查 Python 虚拟环境
if [ ! -d ".venv" ]; then
    osascript -e 'display dialog "未找到 .venv 虚拟环境\n请先运行: python3 -m venv .venv\n然后: source .venv/bin/activate && pip install -e ." buttons {"OK"} with title "SkyGuard 启动失败" with icon caution'
    exit 1
fi

# 检查 skyguard 命令
if [ ! -f ".venv/bin/skyguard" ]; then
    osascript -e 'display dialog "未找到 skyguard 命令\n请先安装依赖: source .venv/bin/activate && pip install -e ." buttons {"OK"} with title "SkyGuard 启动失败" with icon caution'
    exit 1
fi

# 询问端口（默认8000）
PORT=$(osascript -e 'set portNum to text returned of (display dialog "请输入API端口（默认8000）:" default answer "8000" buttons {"启动", "取消"} default button "启动" with title "SkyGuard API 启动器")' 2>/dev/null)

# 用户取消
if [ $? -ne 0 ]; then
    exit 0
fi

# 端口默认值
if [ -z "$PORT" ]; then
    PORT=8000
fi

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    osascript -e "display dialog \"端口 $PORT 已被占用！\n请关闭占用进程后重试。\" buttons {\"OK\"} with title \"SkyGuard 启动失败\" with icon caution"
    exit 1
fi

# 启动服务器
echo "======================================"
echo "  SkyGuard API 启动器"
echo "======================================"
echo "项目目录: $(pwd)"
echo "API地址:  http://localhost:$PORT"
echo "API文档:  http://localhost:$PORT/docs"
echo "======================================"
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动后自动打开浏览器
(sleep 2 && open "http://localhost:$PORT/docs") &

# 启动服务器
.venv/bin/skyguard serve --host 0.0.0.0 --port $PORT
