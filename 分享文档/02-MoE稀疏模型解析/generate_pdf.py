#!/usr/bin/env python3
"""
MoE深度解析文档PDF生成器
将Markdown转换为PDF格式
"""

import os
import sys
from pathlib import Path

def convert_markdown_to_pdf():
    """将Markdown转换为PDF"""

    # 检查依赖
    try:
        import markdown
        from weasyprint import HTML
    except ImportError:
        print("需要安装依赖包：")
        print("pip install markdown weasyprint")
        return False

    # 文件路径
    md_file = Path("60分钟-MoE深度解析.md")
    html_file = Path("moe_presentation_full.html")
    pdf_file = Path("MoE深度解析-60分钟完整版.pdf")

    if not md_file.exists():
        print(f"错误：找不到Markdown文件 {md_file}")
        return False

    print("正在读取Markdown文件...")

    # 读取Markdown内容
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 创建完整的HTML模板
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoE深度解析 - 60分钟完整版</title>
    <style>
        @page {
            size: A4 landscape;
            margin: 2cm;
            @top-center {
                content: "MoE深度解析 | 60分钟完整版";
                font-size: 12px;
                color: #666;
            }
            @bottom-center {
                content: counter(page);
                font-size: 12px;
            }
        }

        body {
            font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
            font-size: 14px;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
        }

        .slide {
            page-break-after: always;
            min-height: 29.7cm;
            padding: 2cm;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        h1 {
            color: #1a5f7a;
            font-size: 36px;
            text-align: center;
            margin-bottom: 1cm;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5cm;
        }

        h2 {
            color: #2c3e50;
            font-size: 28px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin: 1cm 0 0.5cm 0;
        }

        h3 {
            color: #34495e;
            font-size: 22px;
            margin: 0.8cm 0 0.3cm 0;
        }

        pre {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.3;
            overflow-x: auto;
            margin: 0.5cm 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 0.5cm 0;
            font-size: 12px;
        }

        th {
            background: #3498db;
            color: white;
            padding: 8px 12px;
            text-align: left;
            border: 1px solid #2980b9;
        }

        td {
            padding: 6px 12px;
            border: 1px solid #ddd;
        }

        .highlight {
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 0.5cm 0;
            border-radius: 5px;
        }

        .key-point {
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 15px;
            margin: 0.5cm 0;
            border-radius: 5px;
            font-weight: bold;
        }

        .warning {
            background: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 15px;
            margin: 0.5cm 0;
            border-radius: 5px;
        }

        .math-box {
            background: #e7f3ff;
            border-left: 5px solid #0066cc;
            padding: 15px;
            margin: 0.5cm 0;
            border-radius: 5px;
            font-family: 'Cambria Math', serif;
        }

        .slide-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1cm;
            padding-bottom: 0.5cm;
            border-bottom: 2px solid #e9ecef;
        }

        .slide-number {
            color: #6c757d;
            font-size: 14px;
            font-weight: bold;
        }

        .slide-title {
            color: #2c3e50;
            font-size: 24px;
            font-weight: bold;
        }

        .agenda-item {
            display: flex;
            align-items: center;
            margin: 8px 0;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .agenda-time {
            background: #3498db;
            color: white;
            padding: 6px 10px;
            border-radius: 3px;
            margin-right: 12px;
            font-weight: bold;
            min-width: 60px;
            text-align: center;
        }

        .center-content {
            text-align: center;
        }

        .two-columns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1cm;
            margin: 0.5cm 0;
        }
    </style>
</head>
<body>
    {content}
</body>
</html>"""

    # 简单的Markdown到HTML转换
    def markdown_to_html(md_text):
        """将Markdown转换为HTML"""

        # 处理标题
        md_text = md_text.replace('# Part', '<div class="slide"><h1>Part')
        md_text = md_text.replace('---', '</div>\n<div class="slide">')

        # 处理二级标题
        md_text = md_text.replace('## ', '<h2>')
        md_text = md_text.replace('\n', '</h2>\n')

        # 处理代码块
        md_text = md_text.replace('```', '<pre>')

        # 处理表格
        md_text = md_text.replace('|', '</td><td>')

        return md_text

    # 转换内容
    html_content = markdown_to_html(md_content)

    # 完整的HTML文档
    full_html = html_template.format(content=html_content)

    # 保存HTML文件
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"HTML文件已生成: {html_file}")

    # 转换为PDF
    try:
        print("正在生成PDF...")
        HTML(html_file).write_pdf(pdf_file)
        print(f"PDF文件已生成: {pdf_file}")
        return True
    except Exception as e:
        print(f"PDF生成失败: {e}")
        return False

def create_simple_html():
    """创建简单的HTML预览版本"""

    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>MoE深度解析 - 预览版</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        h1 { color: #1a5f7a; }
        .slide { margin: 40px 0; padding: 20px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>MoE深度解析 - 60分钟完整版</h1>
    <p>这是文档的HTML预览版本，完整PDF版本需要安装weasyprint库。</p>

    <div class="slide">
        <h2>文档结构预览</h2>
        <ul>
            <li>Part 1: 历史演进 (12分钟)</li>
            <li>Part 2: 数学原理 (12分钟)</li>
            <li>Part 3: 门控机制 (12分钟)</li>
            <li>Part 4: 工程实现 (10分钟)</li>
            <li>Part 5: 现代架构 (6分钟)</li>
            <li>Part 6: 总结展望 (5分钟)</li>
        </ul>
    </div>

    <div class="slide">
        <h2>核心内容</h2>
        <p>文档包含60+页详细内容，涵盖：</p>
        <ul>
            <li>MoE技术演进历史（1991-2024）</li>
            <li>稀疏计算数学原理</li>
            <li>门控机制四层防护</li>
            <li>分布式训练实现</li>
            <li>Mixtral/DeepSeek案例分析</li>
        </ul>
    </div>

    <p><strong>要生成完整PDF，请运行：</strong> pip install weasyprint && python generate_pdf.py</p>
</body>
</html>"""

    with open("moe_preview.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("预览HTML文件已生成: moe_preview.html")

if __name__ == "__main__":
    print("=== MoE深度解析文档生成器 ===")

    # 尝试生成完整PDF
    if convert_markdown_to_pdf():
        print("✅ 文档生成成功！")
    else:
        print("⚠️ PDF生成失败，创建预览版本...")
        create_simple_html()

    print("\n生成的文件：")
    print("- moe_presentation.html (完整HTML版)")
    print("- moe_preview.html (预览版)")
    print("- MoE深度解析-60分钟完整版.pdf (PDF版)")

    print("\n📖 文档特点：")
    print("• 60分钟完整演讲内容")
    print("• PPT风格的清晰排版")
    print("• 大型ASCII架构图")
    print("• 数学公式与代码示例")
    print("• 问题驱动的历史叙事")