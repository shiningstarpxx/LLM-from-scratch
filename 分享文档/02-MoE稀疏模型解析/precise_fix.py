#!/usr/bin/env python3
"""
精确修复Mermaid语法错误
修复具体的语法问题
"""

import re
from pathlib import Path

def fix_exact_errors(content):
    """修复具体的语法错误"""

    fixed_content = content

    # 修复1: 第348-351行的具体错误
    # 错误的: G[数学表达] --> H[$$g(x) = \text{SparseTopK}(\sigma(W_g x), K)$$]        style G fill:#e3f2fd    style H fill:#e8f5e9```
    # 正确的: 将样式定义放在单独的行，并确保图表完整

    # 查找并修复这种模式
    pattern = r'(G\[数学表达\] --> H\[\$\$g\(x\) = \\text\{SparseTopK\}\(\\sigma\(W_g x\), K\)\$\$\]\s*)(style G fill:#e3f2fd\s*style H fill:#e8f5e9)(```)'

    def fix_exact_error(match):
        before = match.group(1)  # 图表内容
        styles = match.group(2)   # 样式定义
        after = match.group(3)    # ```

        # 将样式定义放在单独的行
        fixed = before + '\n    ' + styles.replace('style G fill:#e3f2fd    style H fill:#e8f5e9', 'style G fill:#e3f2fd\n    style H fill:#e8f5e9')
        fixed += '\n' + after
        return fixed

    fixed_content = re.sub(pattern, fix_exact_error, fixed_content)

    # 修复2: 修复所有类似的样式定义在同一行的问题
    # 查找所有样式定义在同一行的模式
    style_pattern = r'(\]\s*)(style\s+\w+\s+fill:[^\n]*)(\s*```)'

    def fix_all_styles(match):
        before = match.group(1)  # ]后面的空格
        styles = match.group(2)   # 样式定义
        after = match.group(3)    # ```前的空格

        # 将样式定义放在单独的行
        return before + '\n    ' + styles + '\n' + after

    fixed_content = re.sub(style_pattern, fix_all_styles, fixed_content)

    # 修复3: 确保所有图表有完整的结束标记
    # 查找不完整的图表定义
    incomplete_pattern = r'(```mermaid\n)(.*?)(?=\n```|$)'

    def fix_incomplete(match):
        start = match.group(1)  # ```mermaid\n
        content = match.group(2)  # 图表内容

        # 确保内容以换行结束
        if not content.endswith('\n'):
            content += '\n'

        return start + content + '```'

    fixed_content = re.sub(incomplete_pattern, fix_incomplete, fixed_content, flags=re.DOTALL)

    return fixed_content

def validate_fixes(content):
    """验证修复结果"""

    print("=== 验证修复结果 ===\n")

    errors = []

    # 检查1: 是否有样式定义在同一行
    if re.search(r'\]\s*style\s+\w+\s+fill:', content):
        errors.append("❌ 发现样式定义在同一行的错误")

    # 检查2: 是否有不完整的图表定义
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

    input_file = Path("60分钟-MoE深度解析-Mermaid版-语法完全修复.md")

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"📖 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 显示发现的具体问题
    print("\n=== 发现的具体问题 ===\n")

    # 查找第348-351行附近的内容
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if i >= 345 and i <= 355:  # 查看第346-356行
            if '数学表达' in line or 'style G fill' in line:
                print(f"第{i+1}行: {line}")

    # 修复语法错误
    print("\n=== 开始精确修复 ===\n")
    fixed_content = fix_exact_errors(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-最终修复.md")
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