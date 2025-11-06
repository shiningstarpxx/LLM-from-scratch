# 🎯 BPE (Byte Pair Encoding) 可视化演示代码

## 📚 文件说明

本目录包含了BPE算法的形象化演示代码，用于配合Lecture 01的Tokenization学习：

### 📁 文件列表

1. **`bpe_core.py`** - BPE核心算法简洁实现
   - 专注于算法本质，代码清晰易懂
   - 包含多个示例演示
   - 适合理解核心概念

2. **`bpe_visualizer.py`** - BPE算法可视化演示工具
   - 详细的逐步演示过程
   - 支持交互模式和自定义文本
   - 包含统计信息和压缩分析

3. **`README_BPE.md`** - 本说明文档

## 🚀 快速开始

### 🎯 方法一：使用启动脚本（推荐）

最简单的方式是使用我们提供的启动脚本：

```bash
# 进入代码目录
cd /Users/peixingxin/code/spring2025-lectures/学习笔记/01-基础建立/01-Lecture01-Introduction

# 查看使用帮助
./run_bpe_demo.sh help

# 快速演示
./run_bpe_demo.sh core        # 核心算法演示
./run_bpe_demo.sh simple      # 简单可视化
./run_bpe_demo.sh interactive # 交互模式
./run_bpe_demo.sh all         # 运行所有演示
```

### 🔧 方法二：手动运行

如果你想手动控制环境：

```bash
# 进入代码目录
cd /Users/peixingxin/code/spring2025-lectures/学习笔记/01-基础建立/01-Lecture01-Introduction

# 激活虚拟环境
source bpe_env/bin/activate

# 检查Python版本
python --version  # 应该显示 Python 3.13.3
```

#### 运行核心算法演示
```bash
python bpe_core.py
```

#### 运行可视化演示
```bash
# 预设演示
python bpe_visualizer.py --demo simple    # 简单示例
python bpe_visualizer.py --demo chinese   # 中文示例
python bpe_visualizer.py --demo english   # 英文示例

# 交互模式
python bpe_visualizer.py --interactive

# 自定义文本
python bpe_visualizer.py --text "your text here" --merges 5
```

## 📊 演示示例解析

### 示例1: "aaabdaaabac"

**原始字节序列**: `[97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]`

**合并过程**:
1. 步骤1: `[97, 97]` → `[256]` (频率: 4) -> "aa"
2. 步骤2: `[256, 97]` → `[257]` (频率: 2) -> "aaa"
3. 步骤3: `[257, 98]` → `[258]` (频率: 2) -> "aaab"
4. 步骤4: `[258, 100]` → `[259]` (频率: 1) -> "aaabd"
5. 步骤5: `[259, 258]` → `[260]` (频率: 1) -> "aaabdaaab"

**最终序列**: `[260, 97, 99]`

**压缩比**: 11 → 3 (约3.67倍压缩)

### 示例2: "hello world hello"

**关键洞察**:
- 重复的"hello"被高效压缩
- "hello"最终用单个token `[259]` 表示
- 词汇表逐渐构建常用模式

### 示例3: "你好世界"

**UTF-8编码特点**:
- 每个中文字符占用3个字节
- "你" → `[228, 189, 160]`
- BPE能够识别并压缩中文字符模式

## 🎯 学习要点

### 1. BPE算法核心思想
- **自底向上**: 从字节开始逐步合并
- **数据驱动**: 基于语料统计高频模式
- **层次化**: 构建多层次的词汇表示

### 2. 压缩效果分析
```python
# 重复模式效果最佳
"aaaaaa"     → 压缩比: 6.0
"abcabcabc"  → 压缩比: 9.0
"hello world" → 压缩比: 1.83
"你好世界你好" → 压缩比: 4.5
```

### 3. 关键参数
- **`num_merges`**: 控制词汇表大小
- **训练语料**: 决定学到的模式
- **字节编码**: UTF-8作为基础表示

### 4. 实际应用考虑
- **特殊token处理**: 防止重要符号被合并
- **预分词**: GPT-2使用正则表达式预处理
- **效率优化**: 编码时的高效查找算法

## 🔧 代码结构解析

### BPETokenizer类核心方法

```python
class BPETokenizer:
    def train(text, num_merges)     # 训练BPE模型
    def encode(text) -> List[int]    # 文本编码
    def decode(tokens) -> str        # token解码
    def _count_pairs(indices)        # 统计相邻对频率
    def _merge(indices, pair, token) # 执行合并操作
```

### 数据结构

- **`vocab: Dict[int, bytes]`**: token索引到字节的映射
- **`merges: Dict[Tuple[int, int], int]`**: 合并规则字典

## 🤔 思考问题

1. **为什么BPE从字节开始而不是字符？**
   - 字节是文本的最小单位
   - 避免Unicode字符集过大问题
   - 保证所有文本都能表示

2. **BPE的局限性是什么？**
   - 依赖训练语料
   - 可能产生语义不完整的token
   - 对罕见模式处理不佳

3. **如何优化BPE实现？**
   - 更高效的编码算法
   - 特殊token保护机制
   - 并行化训练过程

## 📈 压缩效果对比

| 文本类型 | 原始长度 | BPE长度 | 压缩比 | 特点 |
|---------|---------|---------|--------|------|
| 重复字符 "aaaaaa" | 6 | 1 | 6.0 | 最优压缩 |
| 重复模式 "abcabcabc" | 9 | 1 | 9.0 | 模式识别 |
| 英文 "hello world" | 11 | 6 | 1.83 | 中等压缩 |
| 中文 "你好世界你好" | 18 | 4 | 4.5 | UTF-8压缩 |

## 🎓 学习建议

1. **先运行核心演示**: 理解基本算法流程
2. **尝试可视化工具**: 观察详细的合并过程
3. **实验不同文本**: 测试各种语言和模式
4. **思考优化方向**: 考虑实际应用中的改进

## 🔗 相关资源

- [原始论文](https://arxiv.org/abs/1508.07909) - Sennrich et al. (2016)
- [GPT-2 Tokenizer](https://openai.com/research/better-language-models)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)

---

**💡 提示**: 这些代码是为了教学目的而设计的简化版本，实际的BPE实现会更加复杂和高效。理解核心概念后，可以研究工业级实现如GPT-2 tokenizer。