# 使用指南

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- transformers
- sklearn
- numpy
- pandas

## 安装依赖

```bash
pip install torch transformers scikit-learn numpy pandas
```

## 数据处理流程

1. **原始数据** → **Joern CPG** → **GraphCodeBERT嵌入** → **LIVABLE格式**
2. 每个步骤都有对应的处理工具
3. 支持批量处理和错误恢复

## 训练配置

### 模型参数
- 输入维度: 768
- 隐藏维度: 256
- 类别数: 14
- 批次大小: 16

### SADE参数
- alpha: 1.0 (基础权重)
- beta: 2.0 (类别平衡权重)
- gamma: 0.5 (自适应权重)

## 结果解读

由于数据集存在严重的类别不平衡问题，模型主要预测CWE-119类别。
这是真实世界漏洞检测的常见挑战，需要进一步的数据平衡和特征工程。
