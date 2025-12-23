#!/bin/bash
# 自动提取MoE架构图的完整脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMP_DIR="/tmp/moe_extract_$$"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "=========================================="
echo "MoE架构图自动提取工具"
echo "=========================================="
echo ""

# 下载PDF
echo "步骤1: 下载论文PDF..."
echo ""

download_pdf() {
    local name=$1
    local url=$2
    local output="${TMP_DIR}/${name}.pdf"
    
    echo "  下载 ${name}..."
    if curl -L -f -s "$url" -o "$output" 2>/dev/null; then
        if [ -f "$output" ] && [ -s "$output" ]; then
            echo "  ✓ 下载成功: $(du -h "$output" | cut -f1)"
            echo "$output"
        else
            echo "  ✗ 下载失败：文件为空"
            return 1
        fi
    else
        echo "  ✗ 下载失败：无法访问URL"
        return 1
    fi
}

# 尝试使用Python提取
extract_with_python() {
    local pdf_path=$1
    local output_dir=$2
    
    python3 << PYEOF
import sys
try:
    import fitz
    doc = fitz.open("$pdf_path")
    extracted = []
    
    # 检查前3页
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        images = page.get_images()
        
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # 只保存较大的图片（可能是架构图）
                if len(image_bytes) > 50000:  # 大于50KB
                    filename = f"$output_dir/page{page_num+1}_img{img_index+1}.{image_ext}"
                    with open(filename, "wb") as f:
                        f.write(image_bytes)
                    extracted.append((filename, len(image_bytes)))
            except Exception as e:
                pass
    
    doc.close()
    
    if extracted:
        print(f"提取了 {len(extracted)} 个图片:")
        for fname, size in extracted:
            print(f"  {fname} ({size//1024}KB)")
    else:
        print("未找到图片")
        
except ImportError:
    print("需要安装 PyMuPDF: pip3 install --user pymupdf")
    sys.exit(1)
except Exception as e:
    print(f"提取失败: {e}")
    sys.exit(1)
PYEOF
}

# 主流程
PDFS=(
    "shazeer2017|https://arxiv.org/pdf/1701.06538.pdf"
    "switch2021|https://arxiv.org/pdf/2101.03961.pdf"
    "mixtral2024|https://arxiv.org/pdf/2401.04088.pdf"
)

for pdf_info in "${PDFS[@]}"; do
    IFS='|' read -r name url <<< "$pdf_info"
    echo ""
    echo "处理: $name"
    echo "----------------------------------------"
    
    pdf_path=$(download_pdf "$name" "$url")
    
    if [ -n "$pdf_path" ]; then
        echo ""
        echo "步骤2: 提取图片..."
        extract_with_python "$pdf_path" "$TMP_DIR" || {
            echo ""
            echo "⚠ Python提取失败，请手动提取："
            echo "  1. 打开: $pdf_path"
            echo "  2. 找到架构图（通常在Figure 1或前3页）"
            echo "  3. 截图保存为: ${SCRIPT_DIR}/${name}.png"
        }
    fi
done

echo ""
echo "=========================================="
echo "提取完成！"
echo "=========================================="
echo ""
echo "提取的文件在: $TMP_DIR"
echo ""
echo "下一步："
echo "1. 检查提取的图片文件"
echo "2. 找到对应的架构图"
echo "3. 复制到 ${SCRIPT_DIR}/ 并重命名为："
echo "   - shazeer2017.png"
echo "   - switch2021.png"
echo "   - mixtral2024.png"
echo ""
echo "或者直接打开PDF手动截图："
for pdf_info in "${PDFS[@]}"; do
    IFS='|' read -r name url <<< "$pdf_info"
    if [ -f "${TMP_DIR}/${name}.pdf" ]; then
        echo "  open ${TMP_DIR}/${name}.pdf"
    fi
done


