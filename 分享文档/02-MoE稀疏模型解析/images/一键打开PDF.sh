#!/bin/bash
# 一键下载并打开PDF，方便截图提取架构图

TMP_DIR="/tmp/moe_pdfs_$$"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "正在下载PDF文件..."
echo ""

# 下载PDF
curl -L -s "https://arxiv.org/pdf/1701.06538.pdf" -o shazeer2017.pdf && echo "✓ Shazeer 2017 下载完成" || echo "✗ Shazeer 2017 下载失败"
curl -L -s "https://arxiv.org/pdf/2101.03961.pdf" -o switch2021.pdf && echo "✓ Switch 2021 下载完成" || echo "✗ Switch 2021 下载失败"
curl -L -s "https://arxiv.org/pdf/2401.04088.pdf" -o mixtral2024.pdf && echo "✓ Mixtral 2024 下载完成" || echo "✗ Mixtral 2024 下载失败"

echo ""
echo "正在打开PDF文件..."
echo ""

# 打开PDF（Mac）
if command -v open &> /dev/null; then
    [ -f shazeer2017.pdf ] && open shazeer2017.pdf
    [ -f switch2021.pdf ] && open switch2021.pdf
    [ -f mixtral2024.pdf ] && open mixtral2024.pdf
    
    echo "PDF文件已打开！"
    echo ""
    echo "请按照以下步骤操作："
    echo "1. 在打开的PDF中找到架构图（通常在Figure 1或前3页）"
    echo "2. 使用 Command + Shift + 4 截图"
    echo "3. 或使用预览工具 > 选择工具，框选图片后复制"
    echo "4. 将图片保存到以下目录，并重命名为："
    echo "   $(pwd)/../images/shazeer2017.png"
    echo "   $(pwd)/../images/switch2021.png"
    echo "   $(pwd)/../images/mixtral2024.png"
    echo ""
    echo "PDF文件位置: $TMP_DIR"
else
    echo "PDF文件已下载到: $TMP_DIR"
    echo "请手动打开这些PDF文件提取架构图"
fi
