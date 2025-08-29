#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按CWE类型创建Reveal格式二分类数据集
每个CWE类型单独建文件夹，包含对应的Reveal格式数据
所有标签设为1（因为都是漏洞样本）
"""

import json
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] INFO: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class CWERevealDatasetCreator:
    """按CWE类型创建Reveal格式数据集"""
    
    def __init__(self, input_dir: str = "livable_multiclass_data", output_base: str = "cwe_reveal_datasets"):
        self.input_dir = Path(input_dir)
        self.output_base = Path(output_base)
        self.output_base.mkdir(exist_ok=True)
        
        # 边类型映射
        self.edge_type_names = ["AST", "CFG", "DFG", "CDG"]
        
        logger.info(f"🔄 CWE分类Reveal数据集创建器初始化")
        logger.info(f"📂 输入目录: {self.input_dir}")
        logger.info(f"📂 输出根目录: {self.output_base}")
    
    def convert_structure_to_edges(self, structure: List[List[int]]) -> List[Dict]:
        """将structure信息转换为边列表"""
        edges = []
        
        for edge_info in structure:
            if len(edge_info) >= 3:
                source, target, edge_type = edge_info[0], edge_info[1], edge_info[2]
                
                # 确保边类型有效
                if 0 <= edge_type < len(self.edge_type_names):
                    type_name = self.edge_type_names[edge_type]
                else:
                    type_name = "AST"  # 默认类型
                
                edges.append({
                    "source": source,
                    "target": target,
                    "type": type_name
                })
        
        return edges
    
    def create_reveal_nodes(self, features: List[List[float]]) -> List[Dict]:
        """创建Reveal格式的节点列表"""
        nodes = []
        for i, node_features in enumerate(features):
            nodes.append({
                "id": i,
                "features": node_features,  # 保持768维
                "type": "CODE_NODE"
            })
        return nodes
    
    def convert_sample_to_reveal(self, sample: Dict[str, Any], sample_id: int, cwe_type: str) -> Dict[str, Any]:
        """转换单个样本为Reveal格式（二分类，标签=1）"""
        # 提取数据
        features = sample["features"]
        structure = sample["structure"]
        metadata = sample.get("metadata", {})
        
        # 转换为Reveal格式
        nodes = self.create_reveal_nodes(features)
        edges = self.convert_structure_to_edges(structure)
        
        # 创建Reveal样本（二分类，标签=1表示漏洞）
        reveal_sample = {
            "node_features": [node["features"] for node in nodes],
            "graph": [nodes, edges],
            "targets": [1],  # 二分类标签，1=漏洞
            "metadata": {
                "sample_id": sample_id,
                "cwe_type": cwe_type,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "feature_dim": 768,
                "binary_classification": True,
                "label_meaning": "1=vulnerability, 0=normal",
                **metadata
            }
        }
        
        return reveal_sample
    
    def collect_cwe_data(self) -> Dict[str, Dict[str, List]]:
        """收集按CWE类型分组的数据"""
        logger.info("📋 收集CWE分类数据...")
        
        cwe_data = defaultdict(lambda: {"train": [], "valid": [], "test": []})
        
        for split in ["train", "valid", "test"]:
            input_file = self.input_dir / f"livable_{split}.json"
            if not input_file.exists():
                logger.warning(f"⚠️ 文件不存在: {input_file}")
                continue
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📊 处理 {split} 数据集: {len(data)} 样本")
            
            for sample in data:
                cwe_id = sample['metadata']['cwe_id']
                cwe_data[cwe_id][split].append(sample)
        
        logger.info(f"✅ 收集完成: {len(cwe_data)} 个CWE类型")
        
        # 打印统计信息
        for cwe_id, splits in cwe_data.items():
            total = sum(len(splits[split]) for split in ["train", "valid", "test"])
            logger.info(f"  {cwe_id}: {total}个样本 (训练:{len(splits['train'])}, 验证:{len(splits['valid'])}, 测试:{len(splits['test'])})")
        
        return dict(cwe_data)
    
    def create_cwe_dataset(self, cwe_id: str, cwe_data: Dict[str, List]) -> bool:
        """为单个CWE类型创建Reveal格式数据集"""
        logger.info(f"🔄 创建 {cwe_id} 数据集...")
        
        # 创建CWE专用文件夹
        cwe_dir = self.output_base / cwe_id.lower()
        cwe_dir.mkdir(exist_ok=True)
        
        total_samples = 0
        
        for split in ["train", "valid", "test"]:
            if not cwe_data[split]:
                logger.warning(f"⚠️ {cwe_id} {split} 数据为空，跳过")
                continue
            
            logger.info(f"  处理 {split} 数据: {len(cwe_data[split])} 样本")
            
            # 转换样本
            reveal_samples = []
            for i, sample in enumerate(tqdm(cwe_data[split], desc=f"转换{cwe_id}-{split}")):
                try:
                    reveal_sample = self.convert_sample_to_reveal(sample, i, cwe_id)
                    reveal_samples.append(reveal_sample)
                except Exception as e:
                    logger.warning(f"⚠️ {cwe_id} {split} 样本 {i} 转换失败: {e}")
                    continue
            
            # 保存数据
            if reveal_samples:
                output_file = cwe_dir / f"reveal-{split}-v2.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(reveal_samples, f, ensure_ascii=False, indent=2)
                
                total_samples += len(reveal_samples)
                logger.info(f"  ✅ {split}: {len(reveal_samples)} 样本 -> {output_file}")
        
        # 创建配置文件
        self.create_cwe_config(cwe_dir, cwe_id, total_samples)
        
        logger.info(f"🎯 {cwe_id} 数据集创建完成: {total_samples} 个样本")
        return True
    
    def create_cwe_config(self, cwe_dir: Path, cwe_id: str, total_samples: int):
        """创建CWE数据集配置文件"""
        config = {
            "cwe_type": cwe_id,
            "format": "reveal_binary_classification",
            "feature_dim": 768,
            "num_classes": 2,
            "class_labels": {
                "0": "normal",
                "1": "vulnerability"
            },
            "current_data": {
                "label": 1,
                "description": f"All samples are {cwe_id} vulnerabilities"
            },
            "graph_types": ["AST", "CFG", "DFG", "CDG"],
            "splits": {
                "train": "reveal-train-v2.json",
                "valid": "reveal-valid-v2.json",
                "test": "reveal-test-v2.json"
            },
            "total_samples": total_samples,
            "converted_from": "livable_multiclass_data",
            "description": f"Binary classification dataset for {cwe_id} vulnerability type. All samples have label=1 (vulnerability). To create a complete binary dataset, add normal samples with label=0.",
            "note": "This dataset contains only vulnerability samples. For true binary classification, normal/benign samples with label=0 should be added."
        }
        
        config_file = cwe_dir / "dataset_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 配置文件: {config_file}")
    
    def create_all_cwe_datasets(self) -> bool:
        """创建所有CWE类型的数据集"""
        logger.info("🚀 开始创建所有CWE类型的Reveal数据集...")
        
        # 收集CWE分类数据
        cwe_data = self.collect_cwe_data()
        
        success_count = 0
        
        # 为每个CWE类型创建数据集
        for cwe_id, splits_data in cwe_data.items():
            try:
                if self.create_cwe_dataset(cwe_id, splits_data):
                    success_count += 1
            except Exception as e:
                logger.error(f"❌ {cwe_id} 数据集创建失败: {e}")
                continue
        
        # 创建总体说明文件
        self.create_master_readme(cwe_data)
        
        logger.info(f"🎉 完成! 成功创建 {success_count}/{len(cwe_data)} 个CWE数据集")
        return success_count == len(cwe_data)
    
    def create_master_readme(self, cwe_data: Dict[str, Dict]):
        """创建总体说明文件"""
        readme_content = f"""# CWE分类Reveal格式数据集

## 📊 数据集概述

本数据集将原始的多分类漏洞数据按CWE类型分类，每个CWE类型包含独立的Reveal格式二分类数据集。

## 📁 目录结构

```
cwe_reveal_datasets/
├── README.md                    # 本文件
├── dataset_summary.json         # 数据集统计信息
"""

        total_samples = 0
        for cwe_id, splits_data in sorted(cwe_data.items()):
            cwe_total = sum(len(splits_data[split]) for split in ["train", "valid", "test"])
            total_samples += cwe_total
            
            readme_content += f"├── {cwe_id.lower()}/                    # {cwe_id} 漏洞类型 ({cwe_total}个样本)\n"
            readme_content += f"│   ├── reveal-train-v2.json      # 训练集 ({len(splits_data['train'])}个)\n"
            readme_content += f"│   ├── reveal-valid-v2.json      # 验证集 ({len(splits_data['valid'])}个)\n"
            readme_content += f"│   ├── reveal-test-v2.json       # 测试集 ({len(splits_data['test'])}个)\n"
            readme_content += f"│   └── dataset_config.json       # 配置文件\n"

        readme_content += f"""```

## 🎯 数据特点

- **二分类标签**: 所有样本标签为1（漏洞），需要添加标签为0的正常样本来完成二分类
- **768维特征**: 保持GraphCodeBERT原始嵌入维度
- **图结构**: 包含AST, CFG, DFG, CDG四种边类型
- **总样本数**: {total_samples:,}个漏洞样本
- **CWE类型数**: {len(cwe_data)}种

## 📈 CWE类型分布

| CWE类型 | 训练集 | 验证集 | 测试集 | 总计 | 描述 |
|---------|--------|--------|--------|------|------|
"""

        cwe_descriptions = {
            "CWE-119": "缓冲区溢出",
            "CWE-20": "输入验证不当",
            "CWE-399": "资源管理错误",
            "CWE-125": "越界读取",
            "CWE-264": "权限和访问控制",
            "CWE-200": "信息泄露",
            "CWE-189": "数值错误",
            "CWE-416": "释放后使用",
            "CWE-190": "整数溢出",
            "CWE-362": "竞态条件",
            "CWE-476": "空指针解引用",
            "CWE-787": "越界写入",
            "CWE-284": "访问控制不当",
            "CWE-254": "安全特性"
        }

        for cwe_id, splits_data in sorted(cwe_data.items(), key=lambda x: sum(len(x[1][s]) for s in ["train","valid","test"]), reverse=True):
            train_count = len(splits_data['train'])
            valid_count = len(splits_data['valid'])
            test_count = len(splits_data['test'])
            total_count = train_count + valid_count + test_count
            desc = cwe_descriptions.get(cwe_id, "")
            
            readme_content += f"| {cwe_id} | {train_count:,} | {valid_count:,} | {test_count:,} | {total_count:,} | {desc} |\n"

        readme_content += f"""
## 🔧 使用方法

### 加载单个CWE数据集
```python
import json

# 加载CWE-119缓冲区溢出数据
with open('cwe_reveal_datasets/cwe-119/reveal-train-v2.json', 'r') as f:
    cwe119_train = json.load(f)

print(f"CWE-119训练样本数: {{len(cwe119_train)}}")
print(f"样本格式: {{list(cwe119_train[0].keys())}}")
print(f"标签: {{cwe119_train[0]['targets']}}")  # 应该是[1]
```

### 创建完整二分类数据集
```python
# 注意：当前所有样本标签都是1（漏洞）
# 需要添加标签为0的正常样本来创建完整的二分类数据集

# 示例：合并正常样本
normal_samples = load_normal_samples()  # 需要自己提供
for sample in normal_samples:
    sample['targets'] = [0]  # 设置正常样本标签

# 合并漏洞样本和正常样本
complete_dataset = cwe119_train + normal_samples
```

## 📝 注意事项

1. **标签含义**: 
   - 1 = 漏洞样本
   - 0 = 正常样本（需要自行添加）

2. **数据完整性**: 
   - 当前只包含漏洞样本
   - 要进行真正的二分类，需要添加正常代码样本

3. **特征格式**:
   - `node_features`: 768维特征数组
   - `graph`: [nodes, edges] 图结构
   - `targets`: [1] 二分类标签

## 🎯 应用场景

- 特定CWE类型的漏洞检测
- 二分类漏洞检测模型训练
- 图神经网络代码分析
- 安全研究和基准测试

生成时间: $(date)
"""

        # 保存README
        readme_file = self.output_base / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 保存统计摘要
        summary = {
            "total_samples": total_samples,
            "num_cwe_types": len(cwe_data),
            "cwe_distribution": {
                cwe_id: {
                    "train": len(splits_data['train']),
                    "valid": len(splits_data['valid']),
                    "test": len(splits_data['test']),
                    "total": sum(len(splits_data[s]) for s in ["train", "valid", "test"])
                }
                for cwe_id, splits_data in cwe_data.items()
            },
            "format": "reveal_binary_classification",
            "feature_dim": 768,
            "all_labels": 1,
            "note": "All samples are vulnerability samples with label=1. Normal samples with label=0 need to be added for complete binary classification."
        }
        
        summary_file = self.output_base / "dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 总体说明: {readme_file}")
        logger.info(f"📊 统计摘要: {summary_file}")


def main():
    """主函数"""
    creator = CWERevealDatasetCreator()
    
    success = creator.create_all_cwe_datasets()
    
    if success:
        print(f"\n🎉 所有CWE数据集创建完成!")
        print(f"📁 输出目录: cwe_reveal_datasets/")
        print(f"\n📊 创建的数据集:")
        print(f"  - 14个CWE类型的独立数据集")
        print(f"  - 每个包含train/valid/test三个split")
        print(f"  - 所有样本标签为1（漏洞）")
        print(f"  - 768维GraphCodeBERT特征")
        print(f"  - 标准Reveal格式")
        print(f"\n💡 使用提示:")
        print(f"  - 查看 cwe_reveal_datasets/README.md 了解详情")
        print(f"  - 需要添加标签为0的正常样本来完成二分类")
        print(f"  - 每个CWE文件夹包含独立的数据集")
        return 0
    else:
        print(f"\n❌ 部分数据集创建失败")
        return 1


if __name__ == "__main__":
    exit(main())