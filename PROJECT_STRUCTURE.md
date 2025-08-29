# 项目结构说明

## 📁 目录结构

```
all_vul_multiclass/
├── 📂 training_scripts/          # 训练脚本目录
│   ├── 🔬 核心对比实验脚本 (4个)
│   │   ├── simplified_hetero_livable_fusion.py      # 异构GNN训练集学习
│   │   ├── train_hetero_gnn_sade_simple.py          # 异构GNN+SADE训练集学习  
│   │   ├── train_simple_livable_training_only.py    # 简化LIVABLE训练集学习
│   │   └── train_full_livable_training_only.py      # 完整LIVABLE训练集学习
│   ├── 🔄 原始完整训练脚本
│   │   ├── train_full_livable_pyg.py                # 完整LIVABLE原始训练
│   │   ├── train_simple_livable.py                  # 简化LIVABLE原始训练
│   │   ├── train_with_sade_loss.py                  # SADE损失函数训练
│   │   └── train_heterogeneous_gnn_sade.py          # 异构GNN+SADE完整训练
│
├── 📂 models/                    # 模型实现目录
│   ├── heterogeneous_gnn_pyg.py                     # 异构GNN核心实现
│   └── hetero_gnn_livable_fusion.py                 # GNN+LIVABLE融合模型
│
├── 📂 data/                      # 数据目录
│   ├── livable_multiclass_data/                     # LIVABLE格式多分类数据
│   │   ├── livable_train.json
│   │   ├── livable_valid.json
│   │   └── livable_test.json
│   ├── multiclass_label_mapping.json                # 标签映射文件
│   └── all_vul_full_processed/                      # 完整处理后的数据
│       ├── joern_output/
│       └── graphcodebert_output/
│
├── 📂 results/                   # 训练结果目录
│   ├── 🎯 核心对比实验结果 (4个)
│   │   ├── livable_enhanced_hetero_training_analysis/    # 异构GNN结果
│   │   ├── heterogeneous_gnn_sade_training_analysis/     # 异构GNN+SADE结果
│   │   ├── simple_livable_training_analysis/             # 简化LIVABLE结果  
│   │   └── full_livable_training_analysis/               # 完整LIVABLE结果
│   ├── 📊 历史训练结果
│   │   ├── heterogeneous_gnn_pyg_results/
│   │   ├── multiclass_training_results/
│   │   ├── sade_training_results/
│   │   └── simple_livable_results/
│   └── 📈 性能对比文件
│       ├── model_performance_comparison.csv
│       └── model_performance_comparison.png
│
├── 📂 docs/                      # 文档目录
│   ├── 模型训练文件说明文档.md                        # 详细技术文档
│   ├── comprehensive_model_comparison_report.md     # 模型对比报告
│   └── USAGE_GUIDE.md                              # 使用指南
│
├── 📂 utils/                     # 工具目录
│   ├── create_multiclass_from_scratch.py            # 数据创建工具
│   └── comparative_model_performance_analysis.py    # 性能分析工具
│
├── 📂 tools/                     # 数据处理工具
│   ├── joern_cpg_processor.py                       # Joern CPG处理器
│   └── graphcodebert_processor.py                   # GraphCodeBERT处理器
│
├── 📂 LIVABLE-main/              # 原始LIVABLE框架
├── 📂 model_comparison_archive/   # 历史对比实验存档
├── 📄 CLAUDE.md                  # 项目说明
├── 📄 PROJECT_STRUCTURE.md       # 项目结构文档 (本文件)
└── 📄 README.md                  # 项目介绍
```

## 🎯 核心文件说明

### 训练脚本 (training_scripts/)

#### 核心对比实验 (50轮训练集学习)
- **simplified_hetero_livable_fusion.py**: 异构GNN模型，动态边权重学习
- **train_hetero_gnn_sade_simple.py**: 异构GNN + SADE损失，处理类别不平衡  
- **train_simple_livable_training_only.py**: 简化LIVABLE，轻量级架构
- **train_full_livable_training_only.py**: 完整LIVABLE，原始APPNP算法

#### 完整训练脚本 (包含验证/测试)
- **train_full_livable_pyg.py**: 完整版LIVABLE (APPNP + 早停)
- **train_simple_livable.py**: 简化版LIVABLE (100轮完整训练)
- **train_with_sade_loss.py**: SADE损失函数训练
- **train_heterogeneous_gnn_sade.py**: 异构GNN + SADE完整训练

### 模型实现 (models/)
- **heterogeneous_gnn_pyg.py**: 🏆 核心异构GNN实现
  - `HeterogeneousGNNLayer`: 异构消息传播层
  - `HeterogeneousLIVABLEPygModel`: 完整异构模型
  - `HeterogeneousPygDataset`: 数据集适配器

### 训练结果 (results/)

#### 核心对比实验结果
每个结果目录包含：
- `best_model.pth`: 最佳模型权重
- `results.json`: 详细训练结果和分析

#### 关键对比指标
- **异构GNN**: 边类型重要性权重，图结构建模能力
- **SADE损失**: 类别不平衡处理效果，尾部类别改进
- **LIVABLE架构**: APPNP传播 vs 简单MLP的性能差异

## 🚀 快速开始

### 运行核心对比实验
```bash
cd training_scripts/

# 异构GNN模型
python simplified_hetero_livable_fusion.py

# 异构GNN + SADE损失  
python train_hetero_gnn_sade_simple.py

# 简化LIVABLE
python train_simple_livable_training_only.py

# 完整LIVABLE
python train_full_livable_training_only.py
```

### 查看训练结果
```bash
# 查看结果概览
ls results/

# 查看具体实验结果
cat results/livable_enhanced_hetero_training_analysis/results.json
```

### 阅读技术文档
```bash
# 详细技术说明
cat docs/模型训练文件说明文档.md

# 模型对比报告  
cat docs/comprehensive_model_comparison_report.md
```

## 🔬 实验设计

### 研究问题
1. **异构图神经网络 vs 传统LIVABLE**: 动态边权重学习的效果
2. **SADE损失 vs 交叉熵损失**: 类别不平衡处理能力
3. **简化 vs 完整架构**: 模型复杂度与性能权衡

### 数据集
- **训练集**: 5,317个样本，14个CWE类别
- **类别分布**: 严重不平衡 (CWE-119占71%)
- **特征**: 768维GraphCodeBERT节点嵌入 + 图结构 + 序列特征

### 评估指标
- **主要指标**: 加权F1分数、各类别准确率
- **关键观察**: 边类型重要性权重、尾部类别改进、训练收敛性

## 📊 预期结果

| 模型 | 预期F1 | 主要优势 | 关键创新 |
|------|---------|-----------|----------|
| 异构GNN | 0.18-0.20 | 动态边权重 | 边类型重要性学习 |
| 异构GNN+SADE | 0.19-0.21 | 类别平衡 | 自适应损失函数 |
| 简化LIVABLE | 0.15-0.17 | 轻量级 | 快速训练 |
| 完整LIVABLE | 0.17-0.19 | 原始设计 | APPNP传播算法 |

## 🛠️ 开发历史

该项目经历了多轮迭代优化：
1. **基础LIVABLE实现** → **异构GNN改进** → **SADE损失集成** → **架构对比实验**
2. 从分散的实验脚本整理成结构化的研究项目
3. 重点关注软件漏洞检测中的类别不平衡和图结构建模问题

## 🎯 使用建议

1. **研究重点**: 先运行四个核心对比实验，分析结果差异
2. **实验分析**: 重点关注边类型重要性和尾部类别性能
3. **进一步研究**: 基于结果选择最优架构进行深入优化