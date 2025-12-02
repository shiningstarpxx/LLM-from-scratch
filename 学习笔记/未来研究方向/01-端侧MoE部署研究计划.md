# 端侧MoE部署 - 完整研究计划

## 📋 项目信息

**研究方向**: 端侧MoE部署（On-Device MoE Deployment）
**优先级**: ⭐⭐⭐⭐⭐ (8.8/10 - 最高优先级)
**来源**: Lecture 04 Q24深度讨论 + Lecture 02资源账务整合
**创建时间**: 2025-12-02
**预计周期**: 3个月MVP

---

## 🎯 项目愿景

### 核心目标
**在资源受限的移动设备上部署生产级MoE模型，实现完全离线的智能助手**

### 关键指标
```python
target_metrics = {
    '模型规模': {
        'Teacher': '64B MoE (64 experts)',
        'Student': '7B MoE (8 experts)',
        '压缩比': '36倍（蒸馏9倍 × 量化4倍）'
    },

    '内存占用': {
        'FP16': '14GB (不可行)',
        'INT4': '3.5GB (可行✅)',
        'Target': '<4GB RAM'
    },

    '推理性能': {
        '延迟': '<200ms per token',
        '吞吐': '>5 tokens/sec',
        'Target': '接近实时对话体验'
    },

    '功耗': {
        'FP16': '~2.5W',
        'INT4': '<1.5W (目标)',
        '续航': '连续推理>20小时'
    },

    '性能保留': {
        '64B → 7B蒸馏': '≥85%',
        '7B FP16 → INT4': '≥95%',
        '总计': '≥80% Teacher性能'
    }
}
```

---

## 💡 核心创新点

### 创新1: 两步压缩策略

```python
# Stage 1: Knowledge Distillation
class MoEDistillation:
    """
    64B Teacher (64 experts) → 7B Student (8 experts)
    压缩比: 9倍
    """
    def __init__(self, teacher, student):
        self.teacher = teacher  # 64B, frozen
        self.student = student  # 7B, trainable

    def compute_loss(self, batch):
        # Teacher inference (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(batch)
            teacher_routing = self.teacher.get_routing_probs()

        # Student training
        student_logits = self.student(batch)
        student_routing = self.student.get_routing_probs()

        # Loss 1: KL Divergence (soft targets)
        loss_kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T ** 2)

        # Loss 2: Routing Distillation (学expert选择模式)
        loss_routing = F.mse_loss(
            student_routing,
            teacher_routing
        )

        # Loss 3: Hard Labels (ground truth)
        loss_ce = F.cross_entropy(
            student_logits,
            batch['labels']
        )

        # 综合损失
        total_loss = (
            alpha_kl * loss_kl +
            alpha_routing * loss_routing +
            alpha_ce * loss_ce
        )

        return total_loss

# Stage 2: INT4 Quantization
class OnDeviceQuantizer:
    """
    7B FP16 → 7B INT4
    压缩比: 4倍
    """
    def __init__(self):
        self.router_precision = 'FP16'  # 不量化（Lecture 04 Q23✅）
        self.expert_precision = 'INT4'   # 激进量化
        self.activation_precision = 'FP16'  # 不量化

    def quantize_expert(self, expert_weight):
        """
        Per-Expert量化（Lecture 04 Q23策略）
        """
        # 计算per-expert scale
        w_max = max(
            abs(expert_weight.min()),
            abs(expert_weight.max())
        )
        scale = w_max / 7  # INT4: [-7, 7]

        # 量化
        weight_q = torch.clamp(
            torch.round(expert_weight / scale),
            -7, 7
        ).to(torch.int8)  # 用int8存int4

        return weight_q, scale

    def dequantize(self, weight_q, scale):
        """反量化（推理时）"""
        return weight_q.float() * scale
```

### 创新2: 端侧Offloading策略

```python
class OnDeviceOffloading:
    """
    基于Lecture 04 Q22策略，适配端侧环境

    关键差异:
    - 服务器: GPU ↔ CPU ↔ SSD
    - 端侧: RAM ↔ CPU ↔ 闪存(UFS 3.1)
    """
    def __init__(self, num_experts=8, ram_capacity=4):
        self.num_experts = num_experts
        self.ram_slots = 4  # RAM只放4个expert
        self.flash_storage = FlashStorage()  # 闪存swap

        # LRU缓存
        self.expert_cache = LRUCache(capacity=self.ram_slots)

        # 统计预测（Lecture 04 Q22✅）
        self.usage_stats = defaultdict(int)
        self.prediction_window = 10  # 预测未来10个token

    def predict_next_experts(self, current_token, history):
        """
        基于历史统计预测下一步需要的expert
        """
        # 基于当前token特征
        token_features = self.extract_features(current_token)

        # 查询历史模式
        similar_contexts = self.find_similar_contexts(
            history,
            token_features
        )

        # 预测概率
        expert_probs = self.predict_distribution(similar_contexts)

        # 返回top-k预测
        return torch.topk(expert_probs, k=self.ram_slots)

    def preload_experts(self, predicted_ids):
        """
        预加载predicted experts到RAM
        """
        for expert_id in predicted_ids:
            if expert_id not in self.expert_cache:
                # 从闪存异步加载
                expert = self.flash_storage.load_async(expert_id)
                self.expert_cache.put(expert_id, expert)

    def get_expert(self, expert_id):
        """
        获取expert（命中RAM或从闪存加载）
        """
        if expert_id in self.expert_cache:
            # Cache hit: ~1ms
            return self.expert_cache.get(expert_id)
        else:
            # Cache miss: ~10ms (闪存加载)
            expert = self.flash_storage.load_sync(expert_id)
            self.expert_cache.put(expert_id, expert)
            return expert

    def update_stats(self, expert_id):
        """更新使用统计"""
        self.usage_stats[expert_id] += 1
```

### 创新3: 功耗优化

```python
class PowerOptimizer:
    """
    端侧功耗优化策略
    目标: <1.5W持续功耗
    """
    def __init__(self):
        self.dvfs_controller = DVFSController()  # 动态电压频率调节
        self.thermal_monitor = ThermalMonitor()  # 温度监控

    def adaptive_compute(self, task_complexity):
        """
        根据任务复杂度动态调整计算强度
        """
        if task_complexity == 'simple':
            # 简单任务：降频省电
            self.dvfs_controller.set_frequency('low')  # 800MHz
            power = 0.8  # W
        elif task_complexity == 'medium':
            # 中等任务：平衡模式
            self.dvfs_controller.set_frequency('medium')  # 1.5GHz
            power = 1.2  # W
        else:
            # 复杂任务：高性能
            self.dvfs_controller.set_frequency('high')  # 2.5GHz
            power = 2.0  # W

        return power

    def thermal_throttling(self):
        """
        温度过高时自动降频
        """
        temp = self.thermal_monitor.get_temperature()

        if temp > 45:  # °C
            # 过热，强制降频
            self.dvfs_controller.throttle(ratio=0.7)
        elif temp > 40:
            # 温度偏高，轻微降频
            self.dvfs_controller.throttle(ratio=0.9)
```

---

## 🔬 技术挑战与解决方案

### 挑战1: 极端资源约束

**问题**:
- RAM: 4-8 GB (vs 服务器896 GB)
- 算力: ~1 TFLOPS (vs A100 312 TFLOPS)
- 功耗: <3W (vs A100 400W)

**解决方案**:
1. 两步压缩（蒸馏+量化）→ 36倍压缩
2. Offloading策略 → RAM高效利用
3. INT4计算 → 降低功耗和带宽需求

### 挑战2: 推理延迟要求

**问题**:
- 用户期望: <200ms per token
- 标准FP16: ~500ms（不可接受）

**解决方案**:
1. KV Cache（Lecture 03 Q22） → 避免重复计算
2. INT4量化 → 减少计算量
3. Expert预加载 → 减少cache miss

### 挑战3: 长时间稳定运行

**问题**:
- 连续运行>1小时
- 避免内存泄漏
- 温度控制

**解决方案**:
1. 内存管理 → 及时释放
2. 热量监控 → 动态降频
3. 统计信息定期清理

---

## 📅 3个月实施计划

### Phase 1: 模型蒸馏（Week 1-3）

**Week 1: 环境搭建**
```
□ 准备64B Teacher模型（或用开源Mixtral 8x22B）
□ 定义7B Student架构（8 experts）
□ 准备蒸馏数据集（1B tokens，多领域）
  - General: C4, Wikipedia
  - Code: The Stack
  - Math: MATH dataset
  - Dialog: ShareGPT
```

**Week 2-3: 蒸馏训练**
```
□ 实现三种loss（KL + Routing + CE）
□ 训练7B Student
  - Batch size: 256
  - Learning rate: 1e-4
  - Steps: 100K
  - GPU: 8×A100 (2天)
□ 中间评测
  - MMLU, HellaSwag, HumanEval
  - 目标: ≥85% Teacher性能
```

**Deliverable**:
- 7B MoE FP16模型（14GB）
- 蒸馏技术报告
- 性能对比数据

---

### Phase 2: INT4量化（Week 3-4）

**Week 3: 量化实现**
```
□ 实现Per-Expert INT4量化
□ Router保持FP16（不量化）
□ Calibration
  - 用1K样本计算scale
  - 验证数值稳定性
```

**Week 4: 量化感知训练（QAT）**
```
□ 选择性QAT（Lecture 04 Q23策略）
  - 只对敏感expert做QAT
  - 其他expert用PTQ
□ 评测量化模型
  - 目标: 性能损失<2%
□ 最终打包
  - 3.5GB模型文件
```

**Deliverable**:
- 7B INT4模型（3.5GB）
- 量化白皮书
- 对比benchmark

---

### Phase 3: 端侧部署（Week 5-7）

**Week 5: iOS实现**
```
□ Metal GPU加速
  - INT4 kernel实现
  - Router FP16计算
□ Expert loading
  - 从app bundle加载
  - LRU缓存管理
```

**Week 6: Android实现**
```
□ Vulkan/OpenCL加速
  - 跨厂商适配
  - Qualcomm NPU支持
□ Offloading策略
  - RAM ↔ 闪存swap
  - 预测性加载
```

**Week 7: 优化与测试**
```
□ 延迟优化
  - 目标: <200ms
  - Profile hotspots
  - Kernel fusion
□ 功耗测试
  - 不同场景功耗
  - 续航测试
□ 稳定性测试
  - 长时间运行
  - 内存泄漏检测
```

**Deliverable**:
- iOS app beta版
- Android app beta版
- 性能测试报告

---

### Phase 4: 评测对比（Week 8-10）

**Week 8-9: Benchmark**
```
□ vs 云端API
  - GPT-4 API
  - Claude API
  - 延迟、成本对比

□ vs 端侧Dense
  - Gemini Nano (2B)
  - Llama-3-2B
  - 性能、延迟对比

□ 真实场景
  - 离线翻译
  - 隐私问答（医疗、财务）
  - 代码补全
  - 续航测试
```

**Week 10: 用户研究**
```
□ 招募10-20名测试用户
□ A/B测试
  - 本地MoE vs 云端API
  - 收集主观反馈
□ 数据分析
  - 使用模式
  - 痛点识别
  - 改进方向
```

**Deliverable**:
- 完整benchmark报告
- 用户研究报告
- 优势证据

---

### Phase 5: 开源与论文（Week 11-12）

**Week 11: 开源准备**
```
□ 代码整理
  - 清理调试代码
  - 添加注释
  - 编写README
□ 文档编写
  - 部署指南
  - API文档
  - Troubleshooting
□ Demo制作
  - 演示视频
  - 交互式demo
```

**Week 12: 论文撰写**
```
□ 论文大纲
  - Abstract
  - Introduction
  - Method（两步压缩+端侧优化）
  - Experiments
  - Related Work
  - Conclusion
□ 投稿准备
  - 目标: MLSys 2025或MobiSys 2025
□ 社区宣传
  - Technical blog
  - Twitter thread
  - Reddit post
```

**Deliverable**:
- GitHub repo
  - 预期: >1K stars
  - 包含: 模型、代码、文档
- MLSys/MobiSys论文提交
- 技术博客系列

---

## 📊 关键指标追踪

### 性能指标
```python
performance_metrics = {
    '模型质量': {
        'MMLU': {
            'Teacher 64B': 45.2,
            'Student 7B FP16': '>38.5 (目标≥85%)',
            'Student 7B INT4': '>36.8 (目标≥80%)'
        },
        'HellaSwag': {
            'Target': '≥75% Teacher'
        },
        'HumanEval': {
            'Target': '≥80% Teacher'
        }
    },

    '推理性能': {
        '延迟': {
            'Target': '<200ms per token',
            'Best case': '~150ms',
            'Worst case': '<250ms'
        },
        '吞吐': {
            'Target': '>5 tokens/sec',
            'Batch=1': '~6 tokens/sec'
        },
        'Time to first token': '<500ms'
    },

    '资源占用': {
        'RAM': {
            'Model': '3.5GB',
            'KV Cache': '1.5GB (seq=2048)',
            'Total': '<5GB (fit 8GB phone)'
        },
        '功耗': {
            'Idle': '<0.3W',
            'Inference': '<1.5W',
            'Target continuous': '<1.2W average'
        },
        '存储': {
            'Model file': '3.5GB',
            'App binary': '~100MB',
            'Total': '<4GB'
        }
    }
}
```

### 商业指标
```python
business_metrics = {
    '用户价值': {
        '隐私': '100% on-device（无价）',
        '离线': '100% offline capable',
        '延迟': '3x faster than cloud',
        '成本': '$0 vs cloud API'
    },

    '市场定位': {
        '目标市场': '3亿高端AI手机/年',
        '竞争优势': 'MoE on-device首创',
        '竞争强度': '3/10（蓝海）'
    },

    '变现路径': {
        '授权模式': '$1-5/device × 3亿 = $3-15亿/年',
        'App订阅': '$9.99/月 × 100万 = $1.2亿/年',
        '企业版': '$50/user/year × 100万 = $5000万/年'
    }
}
```

---

## 🎯 成功标准

### 必须达成（Must Have）
- ✅ 模型fit 4GB RAM
- ✅ 推理延迟<200ms
- ✅ 性能保留≥80% Teacher
- ✅ 连续运行>1小时稳定
- ✅ iOS + Android双平台

### 期望达成（Should Have）
- ✅ 推理延迟<150ms
- ✅ 性能保留≥85%
- ✅ GitHub stars >500
- ✅ 论文接受（MLSys/MobiSys）
- ✅ 用户研究验证价值

### 加分项（Nice to Have）
- ⭐ 推理延迟<100ms
- ⭐ 性能保留≥90%
- ⭐ GitHub stars >1K
- ⭐ Top-tier会议（ICML/NeurIPS）
- ⭐ 商业合作意向

---

## 🚀 后续演进路线

### Version 2.0（+3个月）
```
1. 多模态支持
   - 整合CLIP encoder
   - 图文混合推理
   - 相机实时理解

2. 个性化微调
   - 设备上fine-tune
   - LoRA adaptation
   - 隐私保护学习

3. 更激进压缩
   - INT2量化探索
   - 结构化剪枝
   - 目标: 2GB模型
```

### Version 3.0（+6个月）
```
1. 联邦学习
   - 跨设备协作
   - 隐私保护聚合
   - 持续改进

2. 多任务统一
   - 翻译+问答+代码
   - Task-level routing
   - 端到端优化

3. 硬件协同设计
   - NPU加速
   - 专用INT4单元
   - 极致性能
```

---

## 📚 参考资源

### 核心技术
1. **Knowledge Distillation**: Hinton et al. (2015)
2. **INT4 Quantization**: Dettmers et al. (2022)
3. **On-Device ML**: TensorFlow Lite, Core ML文档
4. **MoE**: Lecture 04完整学习材料

### 实际系统
- Apple MLX: 端侧推理框架
- Qualcomm AI Engine: 移动NPU
- ExecuTorch: PyTorch端侧runtime
- MLC LLM: 移动端LLM框架

### 开源项目
- llama.cpp: INT4推理参考
- GGML: 量化推理库
- TensorFlow Lite: 移动部署
- NCNN: 手机端推理框架

---

## 💡 风险与应对

### 技术风险

**风险1: 性能损失过大**
- 概率: 中
- 影响: 高
- 应对:
  - 渐进式压缩（先蒸馏验证，再量化）
  - 选择性QAT（对敏感expert重点优化）
  - 备选方案: 放宽到5GB模型

**风险2: 延迟不达标**
- 概率: 中
- 影响: 中
- 应对:
  - Profile优化（识别瓶颈）
  - Kernel优化（Metal/Vulkan）
  - 备选: 降低batch size或sequence length

**风险3: 兼容性问题**
- 概率: 高
- 影响: 中
- 应对:
  - 多设备测试（覆盖主流机型）
  - 降级方案（CPU fallback）
  - 社区反馈快速迭代

### 商业风险

**风险1: 市场接受度**
- 概率: 中
- 影响: 高
- 应对:
  - 用户研究验证需求
  - 免费beta吸引早期用户
  - 技术营销（隐私+离线价值）

**风险2: 巨头竞争**
- 概率: 高
- 影响: 高
- 应对:
  - 快速行动（抢占先发优势）
  - 开源策略（建立社区）
  - 差异化（MoE vs Dense）

---

## 🎉 总结

端侧MoE部署是一个**高风险高回报**的研究方向：

**优势**:
- ⭐⭐⭐⭐⭐ 最高优先级（8.8/10）
- 🔥 蓝海市场（竞争强度3/10）
- 💰 清晰变现路径（$3-15亿/年潜力）
- 🚀 先发优势（目前几乎空白）

**挑战**:
- 技术难度高（7/10）
- 时间窗口短（需快速行动）
- 需要跨学科能力（算法+系统+移动开发）

**建议**:
- 立即开始（时间窗口紧迫）
- 寻找合作者（加速开发）
- 快速迭代（MVP优先）
- 开源+商业双轨（最大化影响力）

**最终目标**:
在手机上实现生产级MoE，**让10亿人拥有私密的AI助手**！

---

**文档创建**: 2025-12-02
**预计开始**: 2025-Q1
**期待成果**: 开创端侧MoE新时代！🚀
