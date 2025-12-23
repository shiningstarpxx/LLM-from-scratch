#!/usr/bin/env python3
"""
从MoE相关论文PDF中提取架构图
需要安装: pip install pymupdf pillow
"""

import os
import sys
import fitz  # PyMuPDF
from pathlib import Path

def extract_images_from_pdf(pdf_path, output_dir, page_numbers=None):
    """
    从PDF中提取图片
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        page_numbers: 要提取的页码列表（从0开始），None表示提取所有页面
    """
    if not os.path.exists(pdf_path):
        print(f"错误：PDF文件不存在: {pdf_path}")
        return []
    
    doc = fitz.open(pdf_path)
    extracted_files = []
    
    pages_to_extract = page_numbers if page_numbers else range(len(doc))
    
    for page_num in pages_to_extract:
        if page_num >= len(doc):
            continue
            
        page = doc[page_num]
        images = page.get_images()
        
        print(f"页面 {page_num + 1}: 找到 {len(images)} 个图片")
        
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # 生成文件名
                pdf_name = Path(pdf_path).stem
                image_filename = f"{output_dir}/{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
                
                # 保存图片
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                
                extracted_files.append(image_filename)
                print(f"  已保存: {image_filename}")
            except Exception as e:
                print(f"  提取图片 {img_index} 失败: {e}")
    
    doc.close()
    return extracted_files

def main():
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    images_dir = script_dir
    
    # 定义要提取的PDF和对应页面
    pdfs_to_extract = [
        {
            "name": "shazeer2017",
            "url": "https://arxiv.org/pdf/1701.06538.pdf",
            "local_path": "/tmp/shazeer2017.pdf",
            "pages": [0, 1, 2],  # 通常架构图在前几页
            "output_name": "shazeer2017"
        },
        {
            "name": "switch2021",
            "url": "https://arxiv.org/pdf/2101.03961.pdf",
            "local_path": None,
            "pages": [0, 1, 2],
            "output_name": "switch2021"
        },
        {
            "name": "mixtral2024",
            "url": "https://arxiv.org/pdf/2401.04088.pdf",
            "local_path": None,
            "pages": [0, 1, 2],
            "output_name": "mixtral2024"
        }
    ]
    
    print("=" * 60)
    print("MoE架构图提取工具")
    print("=" * 60)
    print()
    
    all_extracted = []
    
    for pdf_info in pdfs_to_extract:
        print(f"\n处理: {pdf_info['name']}")
        print("-" * 60)
        
        pdf_path = pdf_info.get("local_path")
        
        # 如果没有本地路径，尝试下载
        if not pdf_path or not os.path.exists(pdf_path):
            pdf_path = f"/tmp/{pdf_info['name']}.pdf"
            url = pdf_info["url"]
            print(f"正在下载: {url}")
            print(f"保存到: {pdf_path}")
            print("提示：如果下载失败，请手动下载PDF到指定路径")
        
        if os.path.exists(pdf_path):
            files = extract_images_from_pdf(
                pdf_path, 
                str(images_dir),
                pdf_info.get("pages")
            )
            all_extracted.extend(files)
        else:
            print(f"跳过: PDF文件不存在 ({pdf_path})")
            print(f"请手动下载: {pdf_info['url']}")
    
    print("\n" + "=" * 60)
    print("提取完成！")
    print(f"共提取 {len(all_extracted)} 个图片文件")
    print("=" * 60)
    print("\n提示：")
    print("1. 检查提取的图片，找到对应的架构图")
    print("2. 将架构图重命名为标准名称（如 shazeer2017.png）")
    print("3. 删除不需要的图片文件")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("错误：需要安装 PyMuPDF")
        print("安装命令: pip install pymupdf")
        sys.exit(1)


