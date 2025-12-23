#!/usr/bin/env python3
"""
最终手动修复Mermaid语法错误
修复具体的语法问题
"""

import re
from pathlib import Path

def fix_specific_errors(content):
    """修复具体的语法错误"""

    fixed_content = content

    # 修复1: 第355行的重复```标记
    fixed_content = fixed_content.replace('```\n```', '```')

    # 修复2: 第359行的不完整图表定义
    # 错误的: ```mermaidgraph TB
    # 正确的: ```mermaid\ngraph TB
    fixed_content = fixed_content.replace('```mermaidgraph', '```mermaid\ngraph')

    # 修复3: 确保所有样式定义在单独的行
    # 查找所有样式定义在同一行的模式
    style_pattern = r'(\]\s*)(style\s+\w+\s+fill:[^\n]*)(\s*```)'

    def fix_styles(match):
        before = match.group(1)  # ]后面的空格
        styles = match.group(2)   # 样式定义
        after = match.group(3)    # ```前的空格

        # 将样式定义放在单独的行
        return before + '\n    ' + styles + '\n' + after

    fixed_content = re.sub(style_pattern, fix_styles, fixed_content)

    # 修复4: 确保所有图表有完整的结束标记
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

    # 检查1: 是否有重复的```标记
    if '```\n```' in content:
        errors.append("❌ 发现重复的```标记")

    # 检查2: 是否有不完整的图表定义
    if '```mermaidgraph' in content:
        errors.append("❌ 发现不完整的图表定义")

    # 检查3: 是否有样式定义在同一行
    if re.search(r'\]\s*style\s+\w+\s+fill:', content):
        errors.append("❌ 发现样式定义在同一行的错误")

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

    input_file = Path("60分钟-MoE深度解析-Mermaid版-最终修复.md")

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"📖 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 显示发现的具体问题
    print("\n=== 发现的具体问题 ===\n")

    # 查找第355行的问题
    if '```\n```' in content:
        print("❌ 发现重复的```标记")

    # 查找第359行的问题
    if '```mermaidgraph' in content:
        print("❌ 发现不完整的图表定义")

    # 查找样式定义在同一行的问题
    if re.search(r'\]\s*style\s+\w+\s+fill:', content):
        print("❌ 发现样式定义在同一行的错误")

    # 修复语法错误
    print("\n=== 开始修复 ===\n")
    fixed_content = fix_specific_errors(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-语法完全修复.md")
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