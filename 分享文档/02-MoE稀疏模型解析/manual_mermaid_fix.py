#!/usr/bin/env python3
"""
手动修复Mermaid语法错误
修复具体的语法问题
"""

import re
from pathlib import Path

def manual_fix_mermaid(content):
    """手动修复具体的语法错误"""

    fixed_content = content

    # 修复1: 修复第334行的语法错误
    # 错误的: $$g(x) = \text{SparseTopKend}(\sigma(W_g x), K)$$
    # 正确的: $$g(x) = \text{SparseTopK}(\sigma(W_g x), K)$$
    fixed_content = fixed_content.replace(
        '$$g(x) = \\text{SparseTopKend}(\\sigma(W_g x), K)$$',
        '$$g(x) = \\text{SparseTopK}(\\sigma(W_g x), K)$$'
    )

    # 修复2: 检查并修复其他可能的语法错误
    # 检查子图是否缺少结束标记
    subgraph_pattern = r'subgraph[^}]+(?!\n\s*end)'
    matches = re.findall(subgraph_pattern, fixed_content, re.DOTALL)

    for match in matches:
        # 为不完整的子图添加结束标记
        fixed_match = match + '\n    end'
        fixed_content = fixed_content.replace(match, fixed_match)

    # 修复3: 检查数学公式中的换行问题
    # 查找换行的数学公式
    multiline_math_pattern = r'\$\$[^$]*\n[^$]*\$\$'
    multiline_matches = re.findall(multiline_math_pattern, fixed_content)

    for match in multiline_matches:
        # 移除换行符
        fixed_math = match.replace('\n', '').replace('\r', '')
        fixed_content = fixed_content.replace(match, fixed_math)

    # 修复4: 确保所有图表有完整的结束标记
    # 查找不完整的图表定义
    incomplete_pattern = r'```mermaid\n(.*?)(?=\n```|$)'

    def fix_incomplete_chart(match):
        chart_content = match.group(1)
        # 确保图表内容以换行结束
        if not chart_content.endswith('\n'):
            chart_content += '\n'
        return f'```mermaid\n{chart_content}```'

    fixed_content = re.sub(incomplete_pattern, fix_incomplete_chart, fixed_content, flags=re.DOTALL)

    return fixed_content

def validate_fixed_content(content):
    """验证修复后的内容"""

    print("=== 验证修复结果 ===\n")

    # 检查是否还有明显的语法错误
    errors = []

    # 检查1: 是否还有SparseTopKend错误
    if 'SparseTopKend' in content:
        errors.append("发现未修复的SparseTopKend错误")

    # 检查2: 检查不完整的子图
    subgraph_pattern = r'subgraph[^}]+(?!\n\s*end)'
    if re.search(subgraph_pattern, content):
        errors.append("发现不完整的子图定义")

    # 检查3: 检查换行的数学公式
    math_pattern = r'\$\$[^$]*\n[^$]*\$\$'
    if re.search(math_pattern, content):
        errors.append("发现换行的数学公式")

    # 检查4: 检查图表数量
    mermaid_count = content.count('```mermaid')
    print(f"📊 图表总数: {mermaid_count}")

    # 检查5: 检查关键图表是否完整
    key_charts = [
        'timeline',
        'graph TB',
        'graph LR',
        'mindmap',
        'xychart-beta'
    ]

    for chart_type in key_charts:
        count = content.count(chart_type)
        if count > 0:
            print(f"📈 {chart_type}: {count}个")

    if errors:
        print("\n❌ 仍然存在的问题:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ 所有明显的语法错误已修复")
        return True

def main():
    """主函数"""

    input_file = Path("60分钟-MoE深度解析-Mermaid版.md")

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"📖 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 显示发现的具体问题
    print("\n=== 发现的具体问题 ===\n")

    # 查找SparseTopKend错误
    if 'SparseTopKend' in content:
        print("❌ 发现SparseTopKend语法错误")
        # 显示错误位置
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'SparseTopKend' in line:
                print(f"第{i+1}行: {line}")

    # 查找不完整的子图
    subgraph_pattern = r'subgraph[^}]+(?!\n\s*end)'
    if re.search(subgraph_pattern, content):
        print("❌ 发现不完整的子图定义")

    # 查找换行的数学公式
    math_pattern = r'\$\$[^$]*\n[^$]*\$\$'
    if re.search(math_pattern, content):
        print("❌ 发现换行的数学公式")

    # 手动修复
    print("\n=== 开始手动修复 ===\n")
    fixed_content = manual_fix_mermaid(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-最终修复.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✅ 修复后的文件已保存: {output_file}")

    # 验证修复结果
    success = validate_fixed_content(fixed_content)

    if success:
        print("\n🎉 修复成功！")
        print("\n📝 使用建议:")
        print("1. 使用在线Mermaid编辑器验证: https://mermaid.live")
        print("2. 在Marp中测试渲染效果")
        print("3. 检查所有图表在投影时的清晰度")
    else:
        print("\n⚠️ 仍有问题需要进一步检查")

if __name__ == "__main__":
    main()