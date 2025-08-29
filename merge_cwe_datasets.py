#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并CWE分类数据集
将每个CWE类型的train/valid/test合并为一个完整的数据集
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] INFO: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class CWEDatasetMerger:
    """CWE数据集合并器"""
    
    def __init__(self, 
                 input_dir: str = "cwe_reveal_datasets", 
                 output_dir: str = "cwe_reveal_datasets_merged"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔄 CWE数据集合并器初始化")
        logger.info(f"📂 输入目录: {self.input_dir}")
        logger.info(f"📂 输出目录: {self.output_dir}")
    
    def merge_cwe_splits(self, cwe_id: str) -> bool:
        """合并单个CWE类型的train/valid/test数据"""
        logger.info(f"🔄 合并 {cwe_id} 数据集...")
        
        cwe_input_dir = self.input_dir / cwe_id.lower()
        cwe_output_dir = self.output_dir / cwe_id.lower()
        cwe_output_dir.mkdir(exist_ok=True)
        
        # 检查输入文件是否存在
        splits = ["train", "valid", "test"]
        split_files = {}
        total_samples = 0
        
        for split in splits:
            split_file = cwe_input_dir / f"reveal-{split}-v2.json"
            if split_file.exists():
                split_files[split] = split_file
            else:
                logger.warning(f"⚠️ {cwe_id} {split} 文件不存在: {split_file}")
        
        if not split_files:
            logger.error(f"❌ {cwe_id} 没有可合并的数据文件")
            return False
        
        # 合并数据
        merged_data = []
        sample_id = 0
        split_counts = {}
        
        for split in splits:
            if split not in split_files:
                continue
                
            logger.info(f"  📊 处理 {split} 数据...")
            
            with open(split_files[split], 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            
            split_counts[split] = len(split_data)
            
            # 为每个样本添加来源信息并重新编号
            for sample in split_data:
                # 更新sample_id
                sample['metadata']['sample_id'] = sample_id
                sample['metadata']['original_split'] = split
                
                merged_data.append(sample)
                sample_id += 1
            
            logger.info(f"    ✅ {split}: {len(split_data)} 样本")
        
        total_samples = len(merged_data)
        logger.info(f"  📈 合并完成: {total_samples} 个样本")
        
        # 保存合并后的数据集
        output_file = cwe_output_dir / "complete_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  💾 保存至: {output_file}")
        
        # 创建统计信息
        self.create_merged_stats(merged_data, cwe_id, cwe_output_dir, split_counts)
        
        # 更新配置文件
        self.create_merged_config(cwe_id, cwe_output_dir, total_samples, split_counts)
        
        logger.info(f"🎯 {cwe_id} 合并完成: {total_samples} 个样本")
        return True
    
    def create_merged_stats(self, data: List[Dict], cwe_id: str, output_dir: Path, split_counts: Dict):
        """创建合并数据的统计信息"""
        stats = {
            "cwe_type": cwe_id,
            "total_samples": len(data),
            "original_split_distribution": split_counts,
            "feature_stats": {},
            "graph_stats": {},
            "metadata_analysis": {}
        }
        
        if data:
            # 特征统计
            sample_features = [len(sample['node_features']) for sample in data]
            feature_dims = [len(sample['node_features'][0]) if sample['node_features'] else 0 for sample in data]
            
            stats["feature_stats"] = {
                "avg_nodes_per_sample": sum(sample_features) / len(sample_features),
                "max_nodes": max(sample_features),
                "min_nodes": min(sample_features),
                "feature_dim": max(feature_dims) if feature_dims else 0
            }
            
            # 图结构统计
            node_counts = [sample['metadata']['num_nodes'] for sample in data]
            edge_counts = [sample['metadata']['num_edges'] for sample in data]
            
            stats["graph_stats"] = {
                "avg_nodes": sum(node_counts) / len(node_counts),
                "avg_edges": sum(edge_counts) / len(edge_counts),
                "max_nodes": max(node_counts),
                "min_nodes": min(node_counts),
                "max_edges": max(edge_counts),
                "min_edges": min(edge_counts)
            }
            
            # 元数据分析
            original_splits = [sample['metadata']['original_split'] for sample in data]
            split_distribution = dict(Counter(original_splits))
            
            stats["metadata_analysis"] = {
                "split_distribution": split_distribution,
                "binary_classification": data[0]['metadata'].get('binary_classification', True),
                "all_labels_are_1": all(sample['targets'] == [1] for sample in data)
            }
        
        # 保存统计文件
        stats_file = output_dir / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"    📊 统计文件: {stats_file}")
    
    def create_merged_config(self, cwe_id: str, output_dir: Path, total_samples: int, split_counts: Dict):
        """创建合并数据集的配置文件"""
        config = {
            "cwe_type": cwe_id,
            "format": "reveal_binary_classification_merged",
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
            "data_files": {
                "complete_dataset": "complete_dataset.json",
                "statistics": "dataset_statistics.json"
            },
            "total_samples": total_samples,
            "original_split_counts": split_counts,
            "merged_from": "cwe_reveal_datasets",
            "description": f"Merged dataset for {cwe_id} vulnerability type. Contains train/valid/test data combined into one file.",
            "usage": {
                "load_complete": "Load complete_dataset.json for all data",
                "filter_by_split": "Use 'original_split' field to filter by original train/valid/test",
                "custom_split": "Create your own train/valid/test splits from the complete dataset"
            },
            "note": "This is a merged dataset. Use 'original_split' metadata field to identify data source."
        }
        
        config_file = output_dir / "dataset_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"    📋 配置文件: {config_file}")
    
    def get_available_cwe_types(self) -> List[str]:
        """获取所有可用的CWE类型"""
        cwe_types = []
        
        if not self.input_dir.exists():
            logger.error(f"❌ 输入目录不存在: {self.input_dir}")
            return cwe_types
        
        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name.startswith('cwe-'):
                cwe_id = item.name.upper().replace('CWE-', 'CWE-')
                cwe_types.append(cwe_id)
        
        return sorted(cwe_types)
    
    def merge_all_cwe_datasets(self) -> bool:
        """合并所有CWE数据集"""
        logger.info("🚀 开始合并所有CWE数据集...")
        
        cwe_types = self.get_available_cwe_types()
        if not cwe_types:
            logger.error("❌ 未找到可合并的CWE数据集")
            return False
        
        logger.info(f"📋 发现 {len(cwe_types)} 个CWE类型: {cwe_types}")
        
        success_count = 0
        failed_cwe = []
        
        for cwe_id in cwe_types:
            try:
                if self.merge_cwe_splits(cwe_id):
                    success_count += 1
                else:
                    failed_cwe.append(cwe_id)
            except Exception as e:
                logger.error(f"❌ {cwe_id} 合并失败: {e}")
                failed_cwe.append(cwe_id)
        
        # 创建总体说明文件
        self.create_master_readme(cwe_types, success_count, failed_cwe)
        
        if failed_cwe:
            logger.warning(f"⚠️ {len(failed_cwe)} 个CWE合并失败: {failed_cwe}")
        
        logger.info(f"🎉 合并完成! 成功: {success_count}/{len(cwe_types)}")
        return len(failed_cwe) == 0
    
    def create_master_readme(self, cwe_types: List[str], success_count: int, failed_cwe: List[str]):
        """创建总体说明文件"""
        readme_content = f"""# CWE分类合并数据集

## 📊 数据集概述

本数据集是CWE分类Reveal格式数据集的合并版本，将每个CWE类型的train/valid/test三个文件合并为一个完整的数据集。

## 🎯 合并优势

- **数据完整性**: 每个CWE类型的所有数据在一个文件中
- **使用便利**: 无需分别加载多个文件
- **灵活分割**: 可根据需要自定义train/valid/test比例
- **溯源清晰**: 保留`original_split`字段标识数据来源

## 📁 目录结构

```
cwe_reveal_datasets_merged/
├── README.md                    # 本文件
├── merger_summary.json          # 合并统计信息
"""

        # 添加每个CWE类型的目录结构
        for cwe_id in sorted(cwe_types):
            if cwe_id not in failed_cwe:
                readme_content += f"""├── {cwe_id.lower()}/                    # {cwe_id} 漏洞类型
│   ├── complete_dataset.json       # 完整数据集
│   ├── dataset_config.json         # 配置文件
│   └── dataset_statistics.json     # 统计信息
"""

        readme_content += f"""```

## 🔧 使用方法

### 加载完整数据集
```python
import json

# 加载CWE-119完整数据集
with open('cwe_reveal_datasets_merged/cwe-119/complete_dataset.json', 'r') as f:
    cwe119_data = json.load(f)

print(f"CWE-119总样本数: {{len(cwe119_data)}}")
print(f"样本格式: {{list(cwe119_data[0].keys())}}")
```

### 按原始分割过滤数据
```python
# 按原始train/valid/test分割过滤
train_samples = [s for s in cwe119_data if s['metadata']['original_split'] == 'train']
valid_samples = [s for s in cwe119_data if s['metadata']['original_split'] == 'valid'] 
test_samples = [s for s in cwe119_data if s['metadata']['original_split'] == 'test']

print(f"训练: {{len(train_samples)}}, 验证: {{len(valid_samples)}}, 测试: {{len(test_samples)}}")
```

### 自定义数据分割
```python
from sklearn.model_selection import train_test_split

# 自定义70/15/15分割
train_data, temp_data = train_test_split(cwe119_data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"自定义分割 - 训练: {{len(train_data)}}, 验证: {{len(valid_data)}}, 测试: {{len(test_data)}}")
```

## 📊 合并统计

- **成功合并**: {success_count}/{len(cwe_types)} 个CWE类型
- **数据格式**: 标准Reveal格式 + 合并元数据
- **特征维度**: 768维GraphCodeBERT嵌入
- **标签类型**: 二分类（所有样本标签=1，表示漏洞）

"""

        if failed_cwe:
            readme_content += f"""
## ⚠️ 合并失败的CWE类型

以下CWE类型合并失败:
"""
            for cwe_id in failed_cwe:
                readme_content += f"- {cwe_id}\n"

        readme_content += f"""
## 📈 数据特点

- **完整性**: 每个CWE类型包含所有训练、验证、测试数据
- **溯源性**: `original_split`字段标识数据来源
- **一致性**: 所有样本保持Reveal标准格式
- **二分类**: 标签均为1（漏洞），需添加正常样本完成二分类

## 🎯 应用场景

1. **完整CWE分析**: 对特定漏洞类型进行全面分析
2. **自定义数据分割**: 根据研究需求重新分割数据
3. **跨CWE对比**: 比较不同漏洞类型的特征差异
4. **模型训练**: 使用完整数据集训练更稳定的模型

## 💡 使用提示

1. **内存占用**: 大型CWE类型（如CWE-119）数据较大，注意内存使用
2. **数据完整性**: 使用前可通过统计文件验证数据完整性
3. **标签一致性**: 所有样本标签为1，需要添加正常样本进行二分类
4. **分割灵活性**: 可根据研究需求自定义train/valid/test比例

生成时间: {self.get_current_time()}
"""

        # 保存README
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 创建合并摘要
        summary = {
            "total_cwe_types": len(cwe_types),
            "successfully_merged": success_count,
            "failed_cwe_types": failed_cwe,
            "merge_strategy": "train_valid_test_combined",
            "output_format": "single_complete_dataset_per_cwe",
            "preserved_fields": [
                "node_features", "graph", "targets", "metadata"
            ],
            "added_fields": [
                "original_split (in metadata)"
            ],
            "data_features": {
                "feature_dim": 768,
                "format": "reveal_binary_classification_merged",
                "all_labels": 1,
                "graph_types": ["AST", "CFG", "DFG", "CDG"]
            }
        }
        
        summary_file = self.output_dir / "merger_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 总体说明: {readme_file}")
        logger.info(f"📊 合并摘要: {summary_file}")
    
    def get_current_time(self) -> str:
        """获取当前时间字符串"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    merger = CWEDatasetMerger()
    
    success = merger.merge_all_cwe_datasets()
    
    if success:
        print(f"\n🎉 所有CWE数据集合并完成!")
        print(f"📁 输出目录: cwe_reveal_datasets_merged/")
        print(f"\n📊 合并特点:")
        print(f"  ✅ 每个CWE类型合并为单一完整数据集")
        print(f"  ✅ 保留original_split字段标识数据来源")
        print(f"  ✅ 重新编号sample_id保证唯一性")
        print(f"  ✅ 生成详细统计和配置信息")
        print(f"  ✅ 标准Reveal格式 + 768维特征")
        print(f"\n💡 使用建议:")
        print(f"  - 查看 cwe_reveal_datasets_merged/README.md")
        print(f"  - 使用complete_dataset.json加载完整数据")
        print(f"  - 通过original_split字段过滤数据")
        print(f"  - 可自定义train/valid/test分割比例")
        return 0
    else:
        print(f"\n❌ 部分CWE数据集合并失败")
        print(f"请检查日志了解详情")
        return 1


if __name__ == "__main__":
    exit(main())