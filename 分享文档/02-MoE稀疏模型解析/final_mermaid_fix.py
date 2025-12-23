#!/usr/bin/env python3
"""
最终Mermaid语法修复
修复具体的语法错误
"""

import re
from pathlib import Path

def fix_specific_errors(content):
    """修复具体的语法错误"""

    fixed_content = content

    # 修复1: 第342行的SparseTopK错误
    # 错误的: $$g(x) = \text{SparseTopK    end}(\sigma(W_g x), K)$$
    # 正确的: $$g(x) = \text{SparseTopK}(\sigma(W_g x), K)$$
    fixed_content = fixed_content.replace(
        '$$g(x) = \\text{SparseTopK    end}(\\sigma(W_g x), K)$$',
        '$$g(x) = \\text{SparseTopK}(\\sigma(W_g x), K)$$'
    )

    # 修复2: 第357-370行的图表格式问题
    # 检查是否有不完整的图表定义
    # 查找所有Mermaid代码块
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', fixed_content, re.DOTALL)

    for block in mermaid_blocks:
        # 检查是否有不完整的数学公式
        if 'end}' in block and '\text{' in block:
            # 修复数学公式中的end错误
            fixed_block = re.sub(r'\\text\{[^}]*?\s*end\}', lambda m: m.group(0).replace(' end', ''), block)
            fixed_content = fixed_content.replace(block, fixed_block)

    # 修复3: 检查并修复所有不完整的子图
    subgraph_pattern = r'subgraph[^}]+(?!\n\s*end)'

    def fix_subgraph(match):
        subgraph_content = match.group(0)
        # 如果子图没有结束标记，添加一个
        if 'end' not in subgraph_content:
            return subgraph_content + '\n    end'
        return subgraph_content

    fixed_content = re.sub(subgraph_pattern, fix_subgraph, fixed_content, flags=re.DOTALL)

    # 修复4: 检查并修复换行的数学公式
    math_pattern = r'\$\$[^$]*\n[^$]*\$\$'

    def fix_math(match):
        math_content = match.group(0)
        # 移除换行符
        return math_content.replace('\n', '')

    fixed_content = re.sub(math_pattern, fix_math, fixed_content)

    return fixed_content

def validate_fixes(content):
    """验证修复结果"""

    print("=== 验证修复结果 ===\n")

    errors = []

    # 检查1: 是否还有SparseTopK错误
    if 'SparseTopK    end' in content:
        errors.append("❌ 发现未修复的SparseTopK错误")

    # 检查2: 检查不完整的子图
    subgraph_pattern = r'subgraph[^}]+(?!\n\s*end)'
    if re.search(subgraph_pattern, content):
        errors.append("❌ 发现不完整的子图定义")

    # 检查3: 检查换行的数学公式
    math_pattern = r'\$\$[^$]*\n[^$]*\$\$'
    if re.search(math_pattern, content):
        errors.append("❌ 发现换行的数学公式")

    # 检查图表数量
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

    input_file = Path("60分钟-MoE深度解析-Mermaid版.md")

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"📖 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 显示发现的具体问题
    print("\n=== 发现的具体问题 ===\n")

    # 查找第342行附近的内容
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'SparseTopK' in line:
            print(f"第{i+1}行: {line}")
            if i+1 < len(lines):
                print(f"第{i+2}行: {lines[i+1]}")

    # 修复语法错误
    print("\n=== 开始修复 ===\n")
    fixed_content = fix_specific_errors(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-语法修复完成.md")
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