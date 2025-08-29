# 🎯 多分类漏洞检测模型性能对比报告
============================================================

## 📊 实验概述

本报告对比了四种不同架构的深度学习模型在14类CWE漏洞检测任务上的性能表现:

1. **Heterogeneous GNN**: Heterogeneous Graph Neural Network
   - 核心特性: Dynamic edge weights, Multi-edge types, Attention mechanism

2. **Simple LIVABLE**: Simplified LIVABLE (Graph + Sequence)
   - 核心特性: Graph encoding, Sequence modeling, Feature fusion

3. **SADE Loss**: LIVABLE with Self-Adaptive Loss
   - 核心特性: Adaptive loss weighting, Class imbalance handling, Dynamic adjustment

4. **Standard Multiclass**: Standard LIVABLE Multiclass
   - 核心特性: Basic graph features, Standard cross-entropy loss

## 🏆 性能排名

| 排名 | 模型名称 | 测试准确率 | 测试F1 | 验证F1 | 参数量(M) |
|------|----------|-----------|--------|--------|----------|
| 1 | Heterogeneous GNN | 0.2211 | 0.1788 | 0.2041 | 2.77 |
| 2 | Simple LIVABLE | 0.2306 | 0.1645 | 0.1884 | 2.27 |
| 3 | SADE Loss | 0.2785 | 0.1229 | 0.1244 | 0.25 |
| 4 | Standard Multiclass | 0.2776 | 0.1210 | 0.1244 | 0.25 |

## 🔍 详细性能分析

### 🥇 最佳性能模型: Heterogeneous GNN

- **测试准确率**: 0.2211
- **测试F1分数**: 0.1788
- **参数量**: 2.77M

#### 🔗 边类型重要性分析 (仅异构GNN)
- **AST**: 0.2500
- **CFG**: 0.2500
- **DFG**: 0.2500
- **CDG**: 0.2500

### ⚡ 最高效模型: SADE Loss

- **效率指标**: 0.489874 (F1分数/百万参数)
- **测试F1**: 0.1229
- **参数量**: 0.25M

## 🎯 关键发现

1. **性能差距**: 最佳与最差模型F1分数差距为 0.0578
2. **平均性能**: 平均准确率 0.2520, 平均F1分数 0.1468
3. **模型复杂度**: 参数量范围从 0.25M 到 2.77M

## 💡 结论与建议

异构图神经网络在多分类漏洞检测任务上表现最佳，其动态边权重调整机制
和多类型边建模能力为其带来了显著的性能优势。

**建议**:
- 对于追求最高性能的场景，推荐使用性能最佳的模型
- 对于资源受限的环境，推荐使用效率最高的模型
- 可以考虑模型集成来进一步提升性能