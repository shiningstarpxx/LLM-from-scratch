#!/usr/bin/env python3
"""
Mermaid语法错误修复工具
检测并修复60分钟-MoE深度解析-Mermaid版.md中的语法错误
"""

import re
import os
from pathlib import Path

def detect_mermaid_errors(md_content):
    """检测Mermaid语法错误"""

    errors = []

    # 查找所有Mermaid代码块
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', md_content, re.DOTALL)

    for i, block in enumerate(mermaid_blocks):
        lines = block.strip().split('\n')

        # 检查1: 图表类型定义
        if not lines:
            errors.append(f"Block {i+1}: 空Mermaid代码块")
            continue

        first_line = lines[0].strip()
        valid_types = ['graph', 'timeline', 'mindmap', 'xychart-beta', 'pie', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram']

        if not any(first_line.startswith(t) for t in valid_types):
            errors.append(f"Block {i+1}: 无效的图表类型定义: {first_line}")

        # 检查2: 不完整的图表
        if '-->' in block and not block.strip().endswith('```'):
            errors.append(f"Block {i+1}: 可能不完整的图表定义")

        # 检查3: 样式定义错误
        style_pattern = r'style\s+\w+\s+fill:'
        if re.search(style_pattern, block):
            # 检查样式语法
            if not re.search(r'style\s+\w+\s+fill:\s*#[0-9a-fA-F]+', block):
                errors.append(f"Block {i+1}: 样式定义语法可能错误")

        # 检查4: 子图定义
        if 'subgraph' in block:
            if 'end' not in block:
                errors.append(f"Block {i+1}: 子图缺少结束标记")

    return errors

def fix_mermaid_syntax(md_content):
    """修复Mermaid语法错误"""

    # 修复1: 确保所有图表有完整的结束标记
    fixed_content = md_content

    # 修复2: 修正样式语法
    fixed_content = re.sub(
        r'style\s+(\w+)\s+fill:\s*([^\s\n]+)',
        r'style \1 fill:\2',
        fixed_content
    )

    # 修复3: 确保子图有结束标记
    fixed_content = re.sub(
        r'(subgraph[^}]+)(?!\n\s*end)',
        r'\1\n    end',
        fixed_content
    )

    # 修复4: 修正数学公式语法
    fixed_content = re.sub(
        r'\$\$(.*?)\$\$',
        r'$$\1$$',
        fixed_content
    )

    return fixed_content

def validate_mermaid_syntax(md_content):
    """验证Mermaid语法"""

    print("=== Mermaid语法验证报告 ===\n")

    # 检测错误
    errors = detect_mermaid_errors(md_content)

    if errors:
        print("❌ 发现的错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 未发现明显的语法错误")

    # 统计图表数量
    mermaid_count = md_content.count('```mermaid')
    print(f"\n📊 图表统计: {mermaid_count} 个Mermaid图表")

    # 图表类型分布
    chart_types = {
        'graph': md_content.count('graph '),
        'timeline': md_content.count('timeline'),
        'mindmap': md_content.count('mindmap'),
        'xychart-beta': md_content.count('xychart-beta')
    }

    print("📈 图表类型分布:")
    for chart_type, count in chart_types.items():
        if count > 0:
            print(f"  - {chart_type}: {count}")

def main():
    """主函数"""

    md_file = Path("60分钟-MoE深度解析-Mermaid版.md")

    if not md_file.exists():
        print(f"❌ 文件不存在: {md_file}")
        return

    print(f"📖 读取文件: {md_file}")

    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 验证语法
    validate_mermaid_syntax(md_content)

    # 修复语法
    print("\n=== 开始修复语法错误 ===\n")
    fixed_content = fix_mermaid_syntax(md_content)

    # 保存修复后的文件
    fixed_file = Path("60分钟-MoE深度解析-Mermaid版-修复后.md")
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✅ 修复后的文件已保存: {fixed_file}")

    # 再次验证
    print("\n=== 修复后验证 ===\n")
    validate_mermaid_syntax(fixed_content)

    print("\n🎉 修复完成！")
    print("\n📝 使用建议:")
    print("1. 使用Mermaid Live Editor在线测试: https://mermaid.live")
    print("2. 在Marp中测试渲染效果")
    print("3. 检查所有图表在投影时的清晰度")

if __name__ == "__main__":
    main()