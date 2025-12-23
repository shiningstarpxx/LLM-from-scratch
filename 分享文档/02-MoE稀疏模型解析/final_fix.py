#!/usr/bin/env python3
"""
最终手动修复Mermaid语法错误
修复所有剩余的样式定义问题
"""

import re
from pathlib import Path

def fix_all_remaining_errors(content):
    """修复所有剩余的语法错误"""

    fixed_content = content

    # 修复1: 第370-372行的样式定义在同一行
    # 错误的: style A fill:#ffebee    style B fill:#ffebee    style C fill:#c8e6c9 ...
    # 正确的: 将每个样式定义放在单独的行

    # 查找并修复这种模式
    pattern = r'(graph LR[^`]*?)(style A fill:#ffebee\s*style B fill:#ffebee\s*style C fill:#c8e6c9\s*style D fill:#c8e6c9\s*style E fill:#c8e6c9\s*style F fill:#c8e6c9\s*style G fill:#e3f2fd\s*style H fill:#e8f5e9)(```)'

    def fix_styles(match):
        chart_content = match.group(1)  # 图表内容
        styles = match.group(2)         # 样式定义
        end = match.group(3)            # ```

        # 将样式定义放在单独的行
        fixed_styles = '\n    ' + styles.replace('style A fill:#ffebee    style B fill:#ffebee    style C fill:#c8e6c9    style D fill:#c8e6c9    style E fill:#c8e6c9    style F fill:#c8e6c9    style G fill:#e3f2fd    style H fill:#e8f5e9',
                                                   'style A fill:#ffebee\n    style B fill:#ffebee\n    style C fill:#c8e6c9\n    style D fill:#c8e6c9\n    style E fill:#c8e6c9\n    style F fill:#c8e6c9\n    style G fill:#e3f2fd\n    style H fill:#e8f5e9')

        return chart_content + fixed_styles + '\n' + end

    fixed_content = re.sub(pattern, fix_styles, fixed_content, flags=re.DOTALL)

    # 修复2: 修复其他类似的样式定义问题
    # 查找所有样式定义在同一行的模式
    style_pattern = r'(\]\s*)(style\s+\w+\s+fill:[^\n]*)(\s*```)'

    def fix_all_styles(match):
        before = match.group(1)  # ]后面的空格
        styles = match.group(2)   # 样式定义
        after = match.group(3)    # ```前的空格

        # 将样式定义放在单独的行
        return before + '\n    ' + styles + '\n' + after

    fixed_content = re.sub(style_pattern, fix_all_styles, fixed_content)

    # 修复3: 修复第377-378行的样式定义问题
    # 查找并修复这种模式
    pattern2 = r'(graph TD[^`]*?)(style B fill:#c8e6c9\s*style D fill:#ffebee\s*style E fill:#ffebee\s*style F fill:#ffebee\s*style G fill:#ffebee\s*style H fill:#ffebee\s*style J fill:#ffcdd2)(```)'

    def fix_styles2(match):
        chart_content = match.group(1)  # 图表内容
        styles = match.group(2)         # 样式定义
        end = match.group(3)            # ```

        # 将样式定义放在单独的行
        fixed_styles = '\n    ' + styles.replace('style B fill:#c8e6c9    style D fill:#ffebee    style E fill:#ffebee    style F fill:#ffebee    style G fill:#ffebee    style H fill:#ffebee    style J fill:#ffcdd2',
                                                   'style B fill:#c8e6c9\n    style D fill:#ffebee\n    style E fill:#ffebee\n    style F fill:#ffebee\n    style G fill:#ffebee\n    style H fill:#ffebee\n    style J fill:#ffcdd2')

        return chart_content + fixed_styles + '\n' + end

    fixed_content = re.sub(pattern2, fix_styles2, fixed_content, flags=re.DOTALL)

    # 修复4: 修复第383-384行的样式定义问题
    # 查找并修复这种模式
    pattern3 = r'(graph LR[^`]*?)(style A fill:#ffebee\s*style B fill:#ffebee\s*style C fill:#ffebee\s*style D fill:#ffcdd2\s*style E fill:#c8e6c9\s*style F fill:#c8e6c9\s*style G fill:#c8e6c9\s*style H fill:#c8e6c9)(```)'

    def fix_styles3(match):
        chart_content = match.group(1)  # 图表内容
        styles = match.group(2)         # 样式定义
        end = match.group(3)            # ```

        # 将样式定义放在单独的行
        fixed_styles = '\n    ' + styles.replace('style A fill:#ffebee    style B fill:#ffebee    style C fill:#ffebee    style D fill:#ffcdd2    style E fill:#c8e6c9    style F fill:#c8e6c9    style G fill:#c8e6c9    style H fill:#c8e6c9',
                                                   'style A fill:#ffebee\n    style B fill:#ffebee\n    style C fill:#ffebee\n    style D fill:#ffcdd2\n    style E fill:#c8e6c9\n    style F fill:#c8e6c9\n    style G fill:#c8e6c9\n    style H fill:#c8e6c9')

        return chart_content + fixed_styles + '\n' + end

    fixed_content = re.sub(pattern3, fix_styles3, fixed_content, flags=re.DOTALL)

    return fixed_content

def validate_fixes(content):
    """验证修复结果"""

    print("=== 验证修复结果 ===\n")

    errors = []

    # 检查1: 是否有样式定义在同一行
    if re.search(r'graph LR[^`]*style A fill:#ffebee\s*style B fill:#ffebee', content):
        errors.append("❌ 发现样式定义在同一行的错误")

    # 检查2: 是否有样式定义在同一行
    if re.search(r'graph TD[^`]*style B fill:#c8e6c9\s*style D fill:#ffebee', content):
        errors.append("❌ 发现样式定义在同一行的错误")

    # 检查3: 是否有样式定义在同一行
    if re.search(r'graph LR[^`]*style A fill:#ffebee\s*style B fill:#ffebee', content):
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

    input_file = Path("60分钟-MoE深度解析-Mermaid版-最终修复完成.md")

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"📖 读取文件: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 显示发现的具体问题
    print("\n=== 发现的具体问题 ===\n")

    # 查找第370-372行的问题
    if re.search(r'graph LR[^`]*style A fill:#ffebee\s*style B fill:#ffebee', content):
        print("❌ 发现第370-372行样式定义在同一行的错误")

    # 查找第377-378行的问题
    if re.search(r'graph TD[^`]*style B fill:#c8e6c9\s*style D fill:#ffebee', content):
        print("❌ 发现第377-378行样式定义在同一行的错误")

    # 查找第383-384行的问题
    if re.search(r'graph LR[^`]*style A fill:#ffebee\s*style B fill:#ffebee', content):
        print("❌ 发现第383-384行样式定义在同一行的错误")

    # 修复语法错误
    print("\n=== 开始最终修复 ===\n")
    fixed_content = fix_all_remaining_errors(content)

    # 保存修复后的文件
    output_file = Path("60分钟-MoE深度解析-Mermaid版-语法完全修复完成.md")
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