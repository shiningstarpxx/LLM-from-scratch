#!/usr/bin/env python3
"""
手动修复子图定义问题
修复不完整的图表定义
"""

import re
from pathlib import Path

def fix_incomplete_charts(content):
    """修复不完整的图表定义"""

    fixed_content = content

    # 修复1: 第331行的图表定义问题
    # 错误的: G[数学表达] --> H[$$g(x) = \text{SparseTopK}(\sigma(W_g x), K)$$]        style G fill:#e3f2fd    style H fill:#e8f5e9```
    # 正确的: 将样式定义放在单独的行

    # 查找所有类似的错误模式
    pattern = r'(\]\s*)(style\s+\w+\s+fill:[^\n]*)(\s*```)'

    def fix_chart_style(match):
        before = match.group(1)  # ]后面的空格
        styles = match.group(2)   # 样式定义
        after = match.group(3)    # ```前的空格

        # 将样式定义放在单独的行
        return before + '\n    ' + styles + after

    fixed_content = re.sub(pattern, fix_chart_style, fixed_content)

    # 修复2: 确保所有图表有完整的结束标记
    # 查找不完整的图表定义
    incomplete_pattern = r'```mermaid\n(.*?)(?=\n```|$)'

    def fix_incomplete(match):
        chart_content = match.group(1)
        # 确保图表内容以换行结束
        if not chart_content.endswith('\n'):
            chart_content += '\n'
        return f'```mermaid\n{chart_content}```'

    fixed_content = re.sub(incomplete_pattern, fix_incomplete, fixed_content, flags=re.DOTALL)

    # 修复3: 确保所有子图有结束标记
    # 查找不完整的子图
    subgraph_pattern = r'(subgraph[^}]+)(?!\n\s*end)'

    def fix_subgraph(match):
        subgraph_content = match.group(1)
        # 如果子图没有结束标记，添加一个
        if 'end' not in subgraph_content:
            return subgraph_content + '\n    end'
        return subgraph_content

    fixed_content = re.sub(subgraph_pattern, fix_subgraph, fixed_content, flags=re.DOTALL)

    return fixed_content

def validate_fixes(content):
    """验证修复结果"""

    print("=== 验证修复结果 ===\n")

    errors = []

    # 检查1: 是否有样式定义在同一行
    if re.search(r'\]\s*style\s+\w+\s+fill:', content):
        errors.append("❌ 发现样式定义在同一行的错误")

    # 检查2: 是否有不完整的子图
    if re.search(r'subgraph[^}]+(?!\n\s*end)', content):
        errors.append("❌ 发现不完整的子图定义")

    # 检查3: 是否有不完整的图表定义
    if re.search(r'```mermaid\n[^`]*(?<!\n```)', content):
        errors.append("❌ 发现不完整的图表定义")

    # 图表统计
    mermaid_count = content.count('```mermaid')
    print(f"📊 图表总数: {mermaid_count}")

    if errors:
        print("\n❌ 仍然存在的问题:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ 所有语法错误已修复")
        return True

def main():
    """主函数"""

    input_file = Path("60分钟-MoE深度解析-Mermaid版-语法修复完成.md")

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"📖 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 显示发现的具体问题
    print("\n=== 发现的具体问题 ===\n")

    # 查找样式定义在同一行的问题
    style_matches = re.findall(r'\]\s*style\s+\w+\s+fill:[^\n]*', content)
    if style_matches:
        print("❌ 发现样式定义在同一行的错误:")
        for match in style_matches[:3]:  # 只显示前3个例子
            print(f"  - {match.strip()}")

    # 查找不完整的子图
    subgraph_matches = re.findall(r'subgraph[^}]+(?!\n\s*end)', content, re.DOTALL)
    if subgraph_matches:
        print("❌ 发现不完整的子图定义:")
        for match in subgraph_matches[:3]:
            print(f"  - {match.strip()[:100]}...")

    # 修复语法错误
    print("\n=== 开始修复 ===\n")
    fixed_content = fix_incomplete_charts(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-最终修复完成.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✅ 修复后的文件已保存: {output_file}")

    # 验证修复结果
    success = validate_fixes(fixed_content)

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