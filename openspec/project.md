# Project Context

## Purpose
这是一个**CS336 深度学习系统学习项目**，目标是从第一性原理深入理解语言模型的构建过程，重点关注：

1. **系统性理解**: 掌握从tokenization到分布式训练的完整技术栈
2. **性能优化思维**: 培养对计算资源、内存使用和效率的敏感度
3. **批判性学习**: 通过苏格拉底式问答方法建立深度技术洞察
4. **实践验证**: 每个理论概念都通过代码实现验证理解
5. **知识积累**: 系统性记录技术讨论和研究方向，形成可持续学习体系

## Tech Stack

### 核心技术
- **Python 3.8+**: 主要编程语言
- **PyTorch 2.0+**: 深度学习框架
- **CUDA**: GPU编程和加速
- **NumPy**: 数值计算基础
- **Triton**: GPU kernel编写和优化

### NLP工具
- **tiktoken**: OpenAI的tokenizer实现
- **transformers**: HuggingFace的预训练模型库
- **fasttext**: 快速文本分类和向量化
- **kenlm**: 语言模型评估

### 数据处理
- **warcio**: Web Archive文件处理
- **markdownify**: HTML到Markdown转换
- **sqlitedict**: 轻量级持久化存储

### 实验和可视化
- **wandb**: 实验跟踪和可视化
- **matplotlib**: 图表绘制
- **einops**: 张量操作简化

### 前端展示
- **Vite + React**: 交互式讲座内容展示
- **JavaScript/Node.js**: 前端构建和部署

### 分布式和性能
- **torch.distributed**: 分布式训练
- **mmh3, bitarray**: 哈希和去重算法
- **Slurm**: 集群作业调度

## Project Conventions

### Code Style
- **语言偏好**: 主要使用Python，注重可读性和教育价值
- **文档风格**: 详细的代码注释，包含使用示例和数学公式
- **命名规范**:
  - 文件: `lecture_XX.py` 格式的可执行讲座
  - 笔记: 中文Markdown，结构化组织
  - 讨论: `[主题]-深度讨论.md` 格式
- **格式化**: 遵循PEP 8规范，使用4空格缩进

### Architecture Patterns
1. **分层学习体系**:
   ```
   学习笔记/          # 系统化课程笔记
   ├── 01-基础建立/
   │   ├── XX-LectureXX-主题/
   │   │   ├── 00-教学大纲.md
   │   │   ├── 01-深度问答.md
   │   │   ├── 02-深度讨论记录.md
   │   │   └── README.md

   深度讨论/          # 专题技术讨论
   ├── [技术主题]-深度讨论.md
   └── 实践代码和可视化
   ```

2. **可执行讲座模式**:
   - 使用 `execute.py` 运行 `lecture_*.py` 生成trace文件
   - 支持本地执行和远程Slurm集群执行
   - 前端React应用展示交互式内容

3. **学习方法论**:
   - **苏格拉底式问答**: 通过引导性问题深化理解
   - **实践验证**: 每个概念都有对应的代码实现
   - **系统思维**: 从硬件约束到软件设计的全链路思考

### Testing Strategy
- **理论验证**: 通过数学推导验证概念理解
- **代码测试**: 实现关键算法并对比标准库结果
- **性能基准**: 使用FLOP计算和内存分析工具评估
- **实验记录**: 使用wandb跟踪实验结果和可视化

### Git Workflow
- **主分支**: `main` - 稳定的学习进度
- **提交规范**: 描述性commit message，记录学习进展
- **文件组织**:
  - 修改的文件主要是中文Markdown笔记和讨论文档
  - 原始课程代码保持不变
  - 新增实践代码放在对应的学习笔记目录

## Domain Context

### 深度学习系统核心概念
1. **资源账务 (Resource Accounting)**:
   - 精确计算FLOP（浮点运算次数）
   - 内存使用分析（参数、梯度、优化器状态、激活）
   - GPU利用率质量评估（非单纯的利用率数字）

2. **性能优化思维**:
   - **内存墙**: 数据传输成为主要瓶颈
   - **算术强度**: 计算量与内存访问的比率
   - **吞吐-延迟矛盾**: 批处理大小的权衡
   - **并行策略**: 数据并行、张量并行、流水线并行

3. **模型架构设计**:
   - Transformer及其变体（RoPE、GQA等）
   - MoE (Mixture of Experts) 稀疏架构
   - 注意力机制优化（Flash Attention等）

4. **训练技术**:
   - 混合精度训练（FP16/BF16 + FP32主副本）
   - 梯度累积和检查点
   - 分布式训练和通信优化
   - 数据处理和过滤技术

### 关键技术洞察记录
- **7B模型内存**: 精确计算124-172GB（非简化的56GB）
- **KV缓存**: 推理时的内存主要开销
- **浮点数不结合性**: 影响分布式训练的可重复性
- **Emergent abilities**: 高维空间连通性与能力涌现的关系
- **知识蒸馏**: "发现vs模仿"的效率差异

## Important Constraints

### 技术约束
1. **Token限制**: AI助手对话有token预算，需要高效利用
2. **计算资源**: 学习环境主要在macOS，部分实验需要GPU
3. **时间约束**: 12周完整课程学习计划
4. **依赖环境**: 需要Python虚拟环境，部分库（如triton）有平台限制

### 学习约束
1. **连续性**: 每次会话开始需要加载工作记忆和进度
2. **质量要求**: 理论深度 > 快速完成，重视批判性思考
3. **记录要求**: 每次重要讨论必须完整记录，不得遗漏
4. **系统性**: 严格按照依赖关系学习，不跳过基础内容

### 文档约束
1. **语言**: 学习笔记和讨论使用中文，代码注释使用英文
2. **格式**: 统一使用Markdown，支持数学公式和代码块
3. **组织**: 严格的目录结构，便于后续检索和回顾

## External Dependencies

### 课程资源
- **Stanford CS336**: 官方课程材料和讲座视频
- **GitHub仓库**: spring2025-lectures（当前项目）
- **论文库**: arXiv论文的工具支持（arxiv_util.py）

### 开发工具
- **Claude Code CLI**: 企业级AI编程助手
  - 官网: https://codebuddy.woa.com
  - 文档: https://iwiki.woa.com/p/4015845000
- **PyTorch官方文档**: 深度学习框架参考
- **HuggingFace**: 预训练模型和数据集

### 计算资源
- **本地环境**: macOS Darwin 25.0.0
- **远程集群**: Slurm调度系统（通过remote_execute.sh）
- **虚拟环境**: Python .venv用于依赖隔离

### 数据源
- **Common Crawl**: 大规模网络数据
- **Wikipedia**: 高质量文本数据
- **代码数据**: GitHub等代码仓库

## Learning Workflow

### AI助手工作记忆加载
每次会话开始时，AI助手应该：
1. 读取 `AI助手工作记忆.md` 了解项目背景和学习进度
2. 检查 `学习进度.md` 确定当前任务
3. 查看 `深度探索TODO.md` 了解待研究的方向
4. 回顾 `深度讨论/` 目录中的最新技术讨论

### 学习模式
1. **理论先行**: 建立概念框架和系统理解
2. **实践验证**: 通过代码实现验证理论理解
3. **深度思考**: 苏格拉底式问答激发批判性思维
4. **工程视角**: 考虑实际部署和优化需求
5. **完整记录**: 所有重要讨论必须记录到深度讨论目录

### 交互方式
- **引导性问题**: 不直接给答案，通过问题激发思考
- **实时验证**: 鼓励通过代码和数值计算验证理解
- **系统关联**: 将新知识与已有知识体系联系
- **批判性分析**: 鼓励质疑和深入探究技术本质

---

## 项目整体计划和当前进展

### 学习计划总览 (12周完整课程)

**总体结构**: 6个阶段，17个讲座

#### 第1阶段：基础建立 (第1-2周)
- **Lecture 01**: Introduction & Tokenization ✅
- **Lecture 02**: PyTorch Building Blocks & Resource Accounting ✅
- **Lecture 03**: Transformer Architecture ⏸️
- **Lecture 04**: Mixture of Experts (MoE) 🔄

#### 第2阶段：硬件与系统 (第3-4周)
- **Lecture 05**: GPU Architecture
- **Lecture 06**: Efficient Kernels
- **Lecture 07**: Model Parallelism
- **Lecture 08**: Distributed Training

#### 第3阶段：规模与优化 (第5-6周)
- **Lecture 09**: Scaling Laws
- **Lecture 10**: Inference Optimization Part 1
- **Lecture 11**: Inference Optimization Part 2

#### 第4阶段：应用与评估 (第7-8周)
- **Lecture 12**: Model Evaluation
- 实践项目和系统整合

#### 第5阶段：数据工程 (第9-10周)
- **Lecture 13**: Training Data
- **Lecture 14**: Data Processing & Filtering

#### 第6阶段：高级训练技术 (第11-12周)
- **Lecture 15**: RLHF (Reinforcement Learning from Human Feedback)
- **Lecture 16**: RLVR (Reinforcement Learning from Visual Reward)
- **Lecture 17**: RL for Language Models

---

### 当前进度状态

**当前位置**: 第1阶段 - 基础建立
**最新完成**: Lecture 04 Q1-Q12 深度讨论 (2025-11-19)
**整体进度**: 约 20% (2.5/17 讲座)

#### 已完成讲座详情

**Lecture 01 ✅**: Introduction & Tokenization
- BPE算法深度理解和可视化实现
- 4种tokenizer对比分析
- 苏格拉底式深度问答完成

**Lecture 02 ✅**: PyTorch Building Blocks & Resource Accounting
- 7B模型内存精确计算 (124-172GB)
- FLOP计算器和资源账务工具
- 混合精度训练深度分析
- 24个引导性问题全部完成

**Lecture 03 ⏸️**: Transformer Architecture
- 部分理论学习完成
- 待深化实践和问答

**Lecture 04 🔄**: Mixture of Experts (当前重点)
- **Q1-Q6 ✅**: MoE基础概念 (核心动机、专家本质、门控网络、Top-K必要性、参数量与计算量分析)
- **Q7-Q12 ✅**: 门控机制深度解析 (2025-11-19完成)
  - Softmax门控的负载不均衡问题
  - Noisy Top-K门控机制
  - 辅助损失的数学原理
  - Expert Capacity机制
  - 门控的可微分性 (Straight-Through Estimator)
  - Router Z-loss深度解析 (完整数学推导、实验验证、PyTorch实现)
- **Q13-Q18**: 现代MoE架构 (待学习)
- **Q19-Q24**: 训练与优化 (待学习)

#### 关键技术收获
1. **MoE稀疏激活原理**: 用1x计算获得128x模型容量的核心机制
2. **Router Z-loss**: 防止Softmax饱和的约束机制 `L_z = (log Σ exp(logits))²`
3. **负载均衡策略**: 辅助损失 + Expert Capacity的双重保障机制
4. **门控梯度流动**: Top-K离散操作的Straight-Through Estimator方法
5. **专家专业化机制**: Router学习、梯度稀疏性、正反馈循环的系统理解

#### 深度探索方向
- 流式深度学习架构设计
- Emergent abilities的几何理论基础
- 知识蒸馏的信息论极限
- GPU利用率质量评估体系
- 系统性能优化与吞吐延迟平衡
- 数值计算与优化算法前沿

详见: `/深度探索TODO.md`

---

**最后更新**: 2025-11-26
**维护者**: peixingxin + Claude Code
**版本**: v1.1
