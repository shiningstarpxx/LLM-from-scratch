#!/bin/bash

# 🎯 BPE演示快速启动脚本
# 用法: ./run_bpe_demo.sh [demo_type]

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 检查虚拟环境是否存在
VENV_PATH="$SCRIPT_DIR/bpe_env"
if [ ! -d "$VENV_PATH" ]; then
    echo "🔧 创建Python虚拟环境..."
    python3 -m venv "$VENV_PATH"
fi

# 激活虚拟环境
echo "🚀 激活虚拟环境..."
source "$VENV_PATH/bin/activate"

# 根据参数选择演示类型
DEMO_TYPE=${1:-"help"}

case $DEMO_TYPE in
    "core")
        echo "📚 运行BPE核心算法演示..."
        python "$SCRIPT_DIR/bpe_core.py"
        ;;
    "simple")
        echo "🎯 运行简单可视化演示..."
        python "$SCRIPT_DIR/bpe_visualizer.py" --demo simple
        ;;
    "chinese")
        echo "🇨🇳 运行中文示例演示..."
        python "$SCRIPT_DIR/bpe_visualizer.py" --demo chinese
        ;;
    "english")
        echo "🇬🇧 运行英文示例演示..."
        python "$SCRIPT_DIR/bpe_visualizer.py" --demo english
        ;;
    "interactive")
        echo "🎮 启动交互模式..."
        python "$SCRIPT_DIR/bpe_visualizer.py" --interactive
        ;;
    "custom")
        echo "✏️ 自定义文本演示..."
        read -p "请输入文本: " text
        read -p "合并次数 (默认10): " merges
        merges=${merges:-10}
        python "$SCRIPT_DIR/bpe_visualizer.py" --text "$text" --merges $merges
        ;;
    "all")
        echo "🎪 运行所有演示..."
        echo "=== 1. 核心算法演示 ==="
        python "$SCRIPT_DIR/bpe_core.py"
        echo -e "\n=== 2. 简单可视化演示 ==="
        python "$SCRIPT_DIR/bpe_visualizer.py" --demo simple
        ;;
    "help"|"-h"|"--help")
        echo "🎯 BPE演示工具使用说明"
        echo ""
        echo "用法: ./run_bpe_demo.sh [演示类型]"
        echo ""
        echo "可用演示类型:"
        echo "  core        - BPE核心算法演示"
        echo "  simple      - 简单可视化演示 (aaabdaaabac)"
        echo "  chinese     - 中文示例演示"
        echo "  english     - 英文示例演示"
        echo "  interactive - 交互模式"
        echo "  custom      - 自定义文本演示"
        echo "  all         - 运行所有演示"
        echo "  help        - 显示此帮助信息"
        echo ""
        echo "示例:"
        echo "  ./run_bpe_demo.sh core"
        echo "  ./run_bpe_demo.sh interactive"
        echo "  ./run_bpe_demo.sh custom"
        ;;
    *)
        echo "❌ 未知的演示类型: $DEMO_TYPE"
        echo "使用 './run_bpe_demo.sh help' 查看可用选项"
        exit 1
        ;;
esac

echo ""
echo "✅ 演示完成!"
echo "📚 查看README_BPE.md了解更多信息"