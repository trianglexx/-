#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIVABLE格式到Reveal格式转换器
保留768维原始特征，只转换数据结构格式
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] INFO: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class LivableToRevealConverter:
    """LIVABLE到Reveal格式转换器"""
    
    def __init__(self, input_dir: str = "livable_multiclass_data", output_dir: str = "reveal_multiclass_data"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔄 LIVABLE -> Reveal 格式转换器初始化")
        logger.info(f"📂 输入目录: {self.input_dir}")
        logger.info(f"📂 输出目录: {self.output_dir}")
    
    def convert_structure_to_edges(self, structure: List[List[int]]) -> List[Dict]:
        """
        将structure信息转换为边列表
        
        Args:
            structure: 结构信息，每个元素是[source, target, edge_type]
            
        Returns:
            边列表，格式: [{"source": 0, "target": 1, "type": "AST"}, ...]
        """
        edges = []
        
        # 定义边类型映射
        edge_type_names = ["AST", "CFG", "DFG", "CDG"]
        
        for edge_info in structure:
            if len(edge_info) >= 3:
                source, target, edge_type = edge_info[0], edge_info[1], edge_info[2]
                
                # 确保边类型有效
                if 0 <= edge_type < len(edge_type_names):
                    type_name = edge_type_names[edge_type]
                else:
                    type_name = "AST"  # 默认类型
                
                edges.append({
                    "source": source,
                    "target": target,
                    "type": type_name
                })
        
        return edges
    
    def create_reveal_nodes(self, features: List[List[float]]) -> List[Dict]:
        """
        创建Reveal格式的节点列表
        
        Args:
            features: 节点特征矩阵 [n_nodes, 768]
            
        Returns:
            节点列表，格式: [{"id": 0, "features": [...]}, ...]
        """
        nodes = []
        for i, node_features in enumerate(features):
            nodes.append({
                "id": i,
                "features": node_features,  # 保持768维
                "type": "CODE_NODE"  # 统一节点类型
            })
        return nodes
    
    def convert_sample(self, livable_sample: Dict[str, Any], sample_id: int) -> Dict[str, Any]:
        """
        转换单个样本从LIVABLE格式到Reveal格式
        
        Args:
            livable_sample: LIVABLE格式样本
            sample_id: 样本ID
            
        Returns:
            Reveal格式样本
        """
        # 提取LIVABLE数据
        features = livable_sample["features"]  # [n_nodes, 768]
        structure = livable_sample["structure"]  # [[source, target, edge_type], ...]
        label = livable_sample["label"]  # [class_id]
        metadata = livable_sample.get("metadata", {})
        
        # 转换为Reveal格式
        nodes = self.create_reveal_nodes(features)
        edges = self.convert_structure_to_edges(structure)
        
        # 创建Reveal样本
        reveal_sample = {
            "node_features": [node["features"] for node in nodes],  # Reveal标准格式
            "graph": [nodes, edges],  # [节点列表, 边列表]
            "targets": label,  # 保持多分类标签
            "metadata": {
                "sample_id": sample_id,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "feature_dim": 768,  # 保持原始768维
                "converted_from": "livable_multiclass",
                "graph_types": ["AST", "CFG", "DFG", "CDG"],
                **metadata  # 合并原始元数据
            }
        }
        
        return reveal_sample
    
    def convert_dataset(self, split: str) -> bool:
        """
        转换指定数据集分割
        
        Args:
            split: 数据集分割名称 ("train", "valid", "test")
            
        Returns:
            转换是否成功
        """
        input_file = self.input_dir / f"livable_{split}.json"
        output_file = self.output_dir / f"reveal-{split}-v2.json"
        
        if not input_file.exists():
            logger.error(f"❌ 输入文件不存在: {input_file}")
            return False
        
        logger.info(f"🔄 转换 {split} 数据集...")
        
        # 加载LIVABLE数据
        with open(input_file, 'r', encoding='utf-8') as f:
            livable_data = json.load(f)
        
        logger.info(f"📋 加载 {len(livable_data)} 个样本")
        
        # 转换每个样本
        reveal_data = []
        for i, sample in enumerate(tqdm(livable_data, desc=f"转换{split}数据")):
            try:
                reveal_sample = self.convert_sample(sample, i)
                reveal_data.append(reveal_sample)
            except Exception as e:
                logger.warning(f"⚠️ 样本 {i} 转换失败: {e}")
                continue
        
        # 保存Reveal数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reveal_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ {split} 数据集转换完成: {len(reveal_data)} 样本")
        logger.info(f"💾 保存至: {output_file}")
        
        # 输出统计信息
        self.print_statistics(reveal_data, split)
        
        return True
    
    def print_statistics(self, reveal_data: List[Dict], split: str):
        """打印数据集统计信息"""
        if not reveal_data:
            return
        
        # 基本统计
        total_samples = len(reveal_data)
        avg_nodes = np.mean([sample["metadata"]["num_nodes"] for sample in reveal_data])
        avg_edges = np.mean([sample["metadata"]["num_edges"] for sample in reveal_data])
        
        # 标签分布统计  
        from collections import Counter
        all_targets = []
        for sample in reveal_data:
            targets = sample["targets"]
            # 处理嵌套列表情况：[[5]] -> [5] -> 5
            if isinstance(targets, list) and len(targets) > 0:
                if isinstance(targets[0], list):
                    all_targets.extend([item for sublist in targets for item in sublist])
                else:
                    all_targets.extend(targets)
            else:
                all_targets.append(targets)
        target_counts = Counter(all_targets)
        
        logger.info(f"📊 {split} 数据集统计:")
        logger.info(f"  - 总样本数: {total_samples}")
        logger.info(f"  - 平均节点数: {avg_nodes:.1f}")
        logger.info(f"  - 平均边数: {avg_edges:.1f}")
        logger.info(f"  - 特征维度: 768 (保持原始)")
        logger.info(f"  - 标签分布: {dict(list(target_counts.most_common(5)))}")
    
    def convert_all(self) -> bool:
        """转换所有数据集"""
        logger.info("🚀 开始转换所有数据集...")
        
        splits = ["train", "valid", "test"]
        success_count = 0
        
        for split in splits:
            if self.convert_dataset(split):
                success_count += 1
        
        if success_count == len(splits):
            logger.info("🎉 所有数据集转换成功!")
            self.create_config_file()
            return True
        else:
            logger.warning(f"⚠️ 部分转换失败: {success_count}/{len(splits)}")
            return False
    
    def create_config_file(self):
        """创建配置文件"""
        config = {
            "format": "reveal_multiclass",
            "feature_dim": 768,
            "num_classes": 14,
            "graph_types": ["AST", "CFG", "DFG", "CDG"],
            "splits": {
                "train": "reveal-train-v2.json",
                "valid": "reveal-valid-v2.json", 
                "test": "reveal-test-v2.json"
            },
            "converted_from": "livable_multiclass_data",
            "description": "LIVABLE格式转换为Reveal格式，保持768维特征",
            "cwe_classes": [
                "CWE-119", "CWE-20", "CWE-399", "CWE-125", "CWE-264",
                "CWE-200", "CWE-189", "CWE-416", "CWE-190", "CWE-362",
                "CWE-476", "CWE-787", "CWE-284", "CWE-254"
            ]
        }
        
        config_file = self.output_dir / "reveal_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 配置文件已保存: {config_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LIVABLE格式到Reveal格式转换器")
    parser.add_argument("--input", default="livable_multiclass_data", 
                       help="输入目录 (默认: livable_multiclass_data)")
    parser.add_argument("--output", default="reveal_multiclass_data", 
                       help="输出目录 (默认: reveal_multiclass_data)")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="all",
                       help="转换指定分割或全部 (默认: all)")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = LivableToRevealConverter(args.input, args.output)
    
    # 执行转换
    if args.split == "all":
        success = converter.convert_all()
    else:
        success = converter.convert_dataset(args.split)
    
    if success:
        print("\n🎯 转换完成!")
        print(f"📁 输出目录: {args.output}")
        print("\n📊 生成的文件:")
        print("  - reveal-train-v2.json")
        print("  - reveal-valid-v2.json") 
        print("  - reveal-test-v2.json")
        print("  - reveal_config.json")
        print("\n💡 特点:")
        print("  - 保持768维GraphCodeBERT特征")
        print("  - 标准Reveal格式: node_features + graph结构")
        print("  - 完整的14类CWE多分类标签")
        print("  - 兼容现有Reveal处理流水线")
        return 0
    else:
        print("\n❌ 转换失败")
        return 1


if __name__ == "__main__":
    exit(main())