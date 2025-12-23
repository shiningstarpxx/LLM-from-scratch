#!/bin/bash
# MoE架构图快速提取脚本

echo "=========================================="
echo "MoE架构图提取工具"
echo "=========================================="
echo ""

IMAGES_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR="/tmp/moe_images"

mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "正在下载论文PDF..."
echo ""

# 下载Shazeer 2017
echo "1. 下载 Shazeer 2017..."
curl -L -o shazeer2017.pdf "https://arxiv.org/pdf/1701.06538.pdf" 2>/dev/null
if [ -f shazeer2017.pdf ]; then
    echo "   ✓ 下载成功"
else
    echo "   ✗ 下载失败，请手动下载: https://arxiv.org/pdf/1701.06538.pdf"
fi

# 下载Switch Transformer
echo "2. 下载 Switch Transformer..."
curl -L -o switch2021.pdf "https://arxiv.org/pdf/2101.03961.pdf" 2>/dev/null
if [ -f switch2021.pdf ]; then
    echo "   ✓ 下载成功"
else
    echo "   ✗ 下载失败，请手动下载: https://arxiv.org/pdf/2101.03961.pdf"
fi

# 下载Mixtral
echo "3. 下载 Mixtral..."
curl -L -o mixtral2024.pdf "https://arxiv.org/pdf/2401.04088.pdf" 2>/dev/null
if [ -f mixtral2024.pdf ]; then
    echo "   ✓ 下载成功"
else
    echo "   ✗ 下载失败，请手动下载: https://arxiv.org/pdf/2401.04088.pdf"
fi

echo ""
echo "=========================================="
echo "PDF文件已下载到: $TMP_DIR"
echo ""
echo "下一步操作："
echo "1. 打开PDF文件（使用预览或Adobe Reader）"
echo "2. 找到架构图页面（通常在Figure 1或前3页）"
echo "3. 截图或提取图片"
echo "4. 保存为PNG格式到: $IMAGES_DIR"
echo "5. 重命名为对应的文件名："
echo "   - shazeer2017.png"
echo "   - switch2021.png"
echo "   - mixtral2024.png"
echo ""
echo "详细说明请查看: $IMAGES_DIR/提取指南.md"
echo "=========================================="


