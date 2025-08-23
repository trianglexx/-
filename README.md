# 漏洞检测多分类项目 - 核心版本

## 🎯 项目概述

基于Joern CPG + GraphCodeBERT + LIVABLE架构的漏洞检测多分类系统，支持14种CWE类型分类。

## 📁 核心文件结构

```
├── create_multiclass_from_scratch.py    # 多分类数据生成脚本
├── train_multiclass_livable.py          # 多分类模型训练
├── train_with_sade_loss.py             # SADE损失函数训练
├── multiclass_label_mapping.json        # CWE标签映射文件
├── tools/
│   ├── joern_cpg_processor.py           # Joern CPG处理工具
│   └── graphcodebert_processor.py       # GraphCodeBERT处理工具
├── livable_multiclass_data/             # 多分类训练数据
│   ├── livable_train.json              # 训练集 (5,317样本)
│   ├── livable_valid.json              # 验证集 (1,140样本)
│   └── livable_test.json               # 测试集 (1,149样本)
├── all_vul_full_processed/              # 完整处理后数据
│   ├── joern_output/                   # Joern CPG数据
│   └── graphcodebert_output/           # GraphCodeBERT嵌入数据
├── multiclass_training_results/         # 多分类训练结果
└── sade_training_results/              # SADE训练结果
```

## 🚀 快速开始

### 1. 数据生成
```bash
python create_multiclass_from_scratch.py
```

### 2. 模型训练
```bash
# 普通多分类训练
python train_multiclass_livable.py

# SADE损失函数训练
python train_with_sade_loss.py
```

## 📊 数据集信息

- **总样本数**: 7,606个
- **CWE类别**: 14种漏洞类型
- **数据分割**: 70% 训练 / 15% 验证 / 15% 测试
- **特征维度**: 768维 GraphCodeBERT嵌入
- **图结构**: 平均100个节点，1,344条边

## 🏷️ CWE类别映射

| 标签 | CWE类型 | 样本数 | 描述 |
|------|---------|--------|------|
| 0 | CWE-119 | 2,127 | 缓冲区溢出 |
| 1 | CWE-20 | 1,142 | 输入验证不当 |
| 2 | CWE-399 | 736 | 资源管理错误 |
| 3 | CWE-125 | 625 | 越界读取 |
| ... | ... | ... | ... |

## 📈 训练结果

### 普通交叉熵损失
- **测试准确率**: 27.76%
- **测试F1分数**: 12.10%

### SADE损失函数
- **测试准确率**: 27.85% (+0.09%)
- **测试F1分数**: 12.29% (+0.19%)
- **测试精确率**: 10.48% (+2.74%)

## 🔧 技术栈

- **CPG生成**: Joern
- **代码嵌入**: GraphCodeBERT
- **模型架构**: LIVABLE (简化版)
- **损失函数**: 交叉熵 / SADE自适应损失
- **框架**: PyTorch

## 📝 使用说明

详细的API文档和使用指南请参考各脚本内的注释。

## 🎯 性能特点

- ✅ 真实的多分类任务（非虚假100%准确率）
- ✅ 完整的CPG图结构（AST+CFG+DFG+CDG）
- ✅ 自适应损失函数改进
- ✅ 端到端训练流水线
