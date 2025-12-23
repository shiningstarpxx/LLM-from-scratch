#!/usr/bin/env python3
"""
精确修复Mermaid语法错误
修复具体的语法问题
"""

import re
from pathlib import Path

def fix_specific_errors(content):
    """修复具体的语法错误"""

    fixed_content = content

    # 修复1: 修复第335-336行的语法错误
    # 错误的: $$g(x) = \text{SparseTopK
    #         end}(\sigma(W_g x), K)$$
    # 正确的: $$g(x) = \text{SparseTopK}(\sigma(W_g x), K)$$

    fixed_content = re.sub(
        r'\$\$g\(x\) = \\text\{SparseTopK\n\s*end\}\(\\sigma\(W_g x\), K\)\$\$',
        r'$$g(x) = \\text{SparseTopK}(\\sigma(W_g x), K)$$',
        fixed_content
    )

    # 修复2: 确保所有子图有正确的结束标记
    fixed_content = re.sub(
        r'(subgraph[^}]+)(?!\n\s*end)',
        r'\1\n    end',
        fixed_content
    )

    # 修复3: 修复数学公式中的换行问题
    fixed_content = re.sub(
        r'\$\$(.*?)\n\s*(.*?)\$\$',
        r'$$\1\2$$',
        fixed_content
    )

    # 修复4: 确保所有图表有完整的结束标记
    fixed_content = re.sub(
        r'(```mermaid\n.*?)(?=\n```|$)',
        lambda m: m.group(1).rstrip(),
        fixed_content,
        flags=re.DOTALL
    )

    return fixed_content

def validate_fixes(content):
    """验证修复结果"""

    print("=== 验证修复结果 ===\n")

    # 检查是否还有明显的语法错误
    errors = []

    # 检查不完整的数学公式
    math_pattern = r'\$\$[^$]*\n[^$]*\$\$'
    if re.search(math_pattern, content):
        errors.append("发现换行的数学公式")

    # 检查不完整的子图
    subgraph_pattern = r'subgraph[^}]+(?!\n\s*end)'
    if re.search(subgraph_pattern, content):
        errors.append("发现不完整的子图定义")

    # 检查图表数量
    mermaid_count = content.count('```mermaid')
    print(f"📊 图表总数: {mermaid_count}")

    if errors:
        print("❌ 仍然存在的问题:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 所有明显的语法错误已修复")

    return len(errors) == 0

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

    # 查找第335-336行附近的内容
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'SparseTopK' in line:
            print(f"第{i+1}行: {line}")
            if i+1 < len(lines):
                print(f"第{i+2}行: {lines[i+1]}")

    # 修复语法错误
    print("\n=== 开始精确修复 ===\n")
    fixed_content = fix_specific_errors(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-精确修复.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✅ 修复后的文件已保存: {output_file}")

    # 验证修复结果
    success = validate_fixes(fixed_content)

    if success:
        print("\n🎉 修复成功！")
        print("\n📝 下一步建议:")
        print("1. 使用在线Mermaid编辑器验证: https://mermaid.live")
        print("2. 在Marp中测试渲染效果")
        print("3. 检查所有图表在投影时的清晰度")
    else:
        print("\n⚠️ 仍有问题需要手动检查")

if __name__ == "__main__":
    main()