#!/usr/bin/env python3
"""
精确手动修复Mermaid语法错误
修复具体的语法问题
"""

import re
from pathlib import Path

def fix_exact_problems(content):
    """修复具体的语法问题"""

    fixed_content = content

    # 修复1: 修复第370-372行的多余空行
    # 当前:
    # graph LR    A[Dense计算] --> B[FLOPs = 2 × d_model × d_ff]    ...
    #
    #
    #     style A fill:#ffebee
    #     style B fill:#ffebee
    #     ...
    # 修复为: 移除多余空行

    # 查找并修复这种模式
    pattern1 = r'(graph LR[^`]*?)(\\n\\n\\n)(\\s*style A fill:#ffebee)'

    def fix_empty_lines(match):
        chart_content = match.group(1)  # 图表内容
        empty_lines = match.group(2)     # 多余的空行
        styles = match.group(3)          # 样式定义

        # 只保留一个换行符
        return chart_content + '\\n' + styles

    fixed_content = re.sub(pattern1, fix_empty_lines, fixed_content, flags=re.DOTALL)

    # 修复2: 修复第385-386行的多余空行
    pattern2 = r'(graph TD[^`]*?)(\\n\\n\\n)(\\s*style B fill:#c8e6c9)'

    def fix_empty_lines2(match):
        chart_content = match.group(1)  # 图表内容
        empty_lines = match.group(2)     # 多余的空行
        styles = match.group(3)          # 样式定义

        # 只保留一个换行符
        return chart_content + '\\n' + styles

    fixed_content = re.sub(pattern2, fix_empty_lines2, fixed_content, flags=re.DOTALL)

    # 修复3: 修复第401-403行的多余空行
    pattern3 = r'(graph LR[^`]*?)(\\n\\n\\n)(\\s*style A fill:#ffebee)'

    def fix_empty_lines3(match):
        chart_content = match.group(1)  # 图表内容
        empty_lines = match.group(2)     # 多余的空行
        styles = match.group(3)          # 样式定义

        # 只保留一个换行符
        return chart_content + '\\n' + styles

    fixed_content = re.sub(pattern3, fix_empty_lines3, fixed_content, flags=re.DOTALL)

    return fixed_content

def validate_fixes(content):
    """验证修复结果"""

    print("=== 验证修复结果 ===\\n")

    errors = []

    # 检查1: 是否有样式定义在同一行
    if re.search(r'graph LR[^`]*style A fill:#ffebee\\s*style B fill:#ffebee', content):
        errors.append("❌ 发现样式定义在同一行的错误")

    # 检查2: 是否有样式定义在同一行
    if re.search(r'graph TD[^`]*style B fill:#c8e6c9\\s*style D fill:#ffebee', content):
        errors.append("❌ 发现样式定义在同一行的错误")

    # 检查3: 是否有样式定义在同一行
    if re.search(r'graph LR[^`]*style A fill:#ffebee\\s*style B fill:#ffebee', content):
        errors.append("❌ 发现样式定义在同一行的错误")

    # 图表统计
    mermaid_count = content.count('```mermaid')
    print(f"📊 图表总数: {mermaid_count}")

    if errors:
        print("\\n❌ 仍然存在的问题:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\\n✅ 所有语法错误已修复")
        return True

def main():
    """主函数"""

    input_file = Path("60分钟-MoE深度解析-Mermaid版-语法完全修复完成.md")

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"📖 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 显示发现的具体问题
    print("\\n=== 发现的具体问题 ===\\n")

    # 查找样式定义在同一行的问题
    if re.search(r'graph LR[^`]*style A fill:#ffebee\\s*style B fill:#ffebee', content):
        print("❌ 发现第370-372行样式定义在同一行的错误")

    if re.search(r'graph TD[^`]*style B fill:#c8e6c9\\s*style D fill:#ffebee', content):
        print("❌ 发现第385-386行样式定义在同一行的错误")

    if re.search(r'graph LR[^`]*style A fill:#ffebee\\s*style B fill:#ffebee', content):
        print("❌ 发现第401-403行样式定义在同一行的错误")

    # 修复语法错误
    print("\\n=== 开始精确修复 ===\\n")
    fixed_content = fix_exact_problems(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-最终修复完成.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_content)

    print(f"✅ 修复后的文件已保存: {output_file}")

    # 验证修复结果
    success = validate_fixes(fixed_content)

    if success:
        print("\\n🎉 修复成功！")
        print("\\n📝 使用建议:")
        print("1. 使用在线Mermaid编辑器验证: https://mermaid.live")
        print("2. 在Marp中测试渲染效果")
        print("3. 检查所有图表在投影时的清晰度")
    else:
        print("\\n⚠️ 仍有问题需要进一步检查")

if __name__ == "__main__":
    main()