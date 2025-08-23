#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始数据直接创建多分类数据集
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path

def create_multiclass_dataset_direct():
    """直接从原始数据创建多分类数据集"""
    
    print("🎯 直接创建CWE多分类数据集")
    print("=" * 60)
    
    # 1. 加载原始数据
    print("📥 加载原始数据...")
    df = pd.read_csv('../all_vul.csv')
    
    # 2. 分析CWE分布
    print("\n📊 CWE分布分析:")
    cwe_counts = df['CWE ID'].value_counts()
    print(f"总CWE类型数: {len(cwe_counts)}")
    
    # 3. 选择主要CWE类型（样本数>=100）
    min_samples = 100
    major_cwes = cwe_counts[cwe_counts >= min_samples]
    print(f"选择样本数>={min_samples}的CWE类型: {len(major_cwes)}个")
    
    # 过滤数据
    df_filtered = df[df['CWE ID'].isin(major_cwes.index)].copy()
    print(f"过滤后样本数: {len(df_filtered)}")
    
    # 4. 创建标签映射
    cwe_to_label = {cwe: i for i, cwe in enumerate(major_cwes.index)}
    label_to_cwe = {i: cwe for cwe, i in cwe_to_label.items()}
    
    print(f"\n🏷️ 标签映射 ({len(major_cwes)}个类别):")
    for i, cwe in label_to_cwe.items():
        count = major_cwes[cwe]
        print(f"   标签 {i}: {cwe} ({count} 样本)")
    
    # 5. 添加数字标签
    df_filtered['multiclass_label'] = df_filtered['CWE ID'].map(cwe_to_label)
    
    # 6. 保存标签映射
    mapping_info = {
        'cwe_to_label': cwe_to_label,
        'label_to_cwe': label_to_cwe,
        'num_classes': len(major_cwes),
        'min_samples_threshold': min_samples,
        'total_samples': len(df_filtered),
        'class_distribution': {str(i): int(major_cwes[cwe]) for i, cwe in label_to_cwe.items()}
    }
    
    with open('multiclass_label_mapping.json', 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    print(f"\n💾 标签映射已保存: multiclass_label_mapping.json")
    
    return df_filtered, mapping_info

def create_simplified_multiclass_data(df_filtered, mapping_info):
    """创建简化的多分类数据"""
    
    print("\n🔄 创建简化的多分类训练数据...")
    
    # 按标签分层抽样
    samples_by_label = {}
    for _, row in df_filtered.iterrows():
        label = row['multiclass_label']
        if label not in samples_by_label:
            samples_by_label[label] = []
        samples_by_label[label].append(row)
    
    print(f"   各类别样本数:")
    for label in sorted(samples_by_label.keys()):
        cwe_id = mapping_info['label_to_cwe'][label]
        count = len(samples_by_label[label])
        print(f"     标签 {label} ({cwe_id}): {count} 样本")
    
    # 分层分割数据
    train_data, valid_data, test_data = [], [], []
    
    for label, samples in samples_by_label.items():
        if len(samples) < 3:
            # 样本太少，全部放入训练集
            train_data.extend(samples)
        else:
            # 分层分割
            train_samples, temp_samples = train_test_split(
                samples, test_size=0.3, random_state=42
            )
            if len(temp_samples) >= 2:
                valid_samples, test_samples = train_test_split(
                    temp_samples, test_size=0.5, random_state=42
                )
            else:
                valid_samples = temp_samples[:1] if temp_samples else []
                test_samples = temp_samples[1:] if len(temp_samples) > 1 else []
            
            train_data.extend(train_samples)
            valid_data.extend(valid_samples)
            test_data.extend(test_samples)
    
    splits = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }
    
    print(f"\n📊 分割结果:")
    for split_name, split_data in splits.items():
        labels = [s['multiclass_label'] for s in split_data]
        label_counts = Counter(labels)
        print(f"   {split_name}: {len(split_data)} 样本")
        print(f"     标签分布: {dict(sorted(label_counts.items()))}")
    
    # 转换为简化的LIVABLE格式
    output_dir = Path("livable_multiclass_data")
    output_dir.mkdir(exist_ok=True)
    
    for split_name, split_data in splits.items():
        print(f"\n🔄 转换 {split_name} 数据...")
        
        livable_samples = []
        for i, row in enumerate(split_data):
            if i % 1000 == 0 and i > 0:
                print(f"   进度: {i}/{len(split_data)}")
            
            livable_sample = create_simplified_sample(row)
            if livable_sample:
                livable_samples.append(livable_sample)
        
        # 保存
        output_file = output_dir / f"livable_{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(livable_samples, f, indent=2, ensure_ascii=False)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ {split_name} 转换完成: {len(livable_samples)} 样本 -> {output_file} ({file_size_mb:.1f}MB)")
    
    return output_dir

def create_simplified_sample(row):
    """创建简化的样本（基于代码文本特征）"""
    
    try:
        # 获取基本信息
        multiclass_label = row['multiclass_label']
        func_code = str(row.get('processed_func', ''))
        cwe_id = row['CWE ID']
        
        if not func_code or func_code == 'nan':
            return None
        
        # 创建简化的代码特征
        code_features = create_code_features(func_code)
        
        # 创建简化的图结构
        graph_data = create_simple_graph(func_code, code_features)
        
        # 构建LIVABLE样本
        livable_sample = {
            'features': graph_data['node_features'],
            'structure': graph_data['edges'],
            'label': [[int(multiclass_label)]],
            'sequence': graph_data['sequence'],
            'metadata': {
                'id': int(row.get('index', 0)),
                'cwe_id': cwe_id,
                'multiclass_label': int(multiclass_label),
                'num_nodes': len(graph_data['node_features']),
                'num_edges': len(graph_data['edges']),
                'code_length': len(func_code)
            }
        }
        
        return livable_sample
        
    except Exception as e:
        print(f"⚠️ 样本转换失败: {e}")
        return None

def create_code_features(func_code):
    """从代码文本创建特征向量"""
    
    # 基础代码统计特征
    features = []
    
    # 长度特征
    features.append(len(func_code) / 1000.0)  # 代码长度（归一化）
    features.append(len(func_code.split()) / 100.0)  # 词汇数量
    features.append(len(func_code.split('\\n')) / 50.0)  # 行数
    
    # 关键词特征
    keywords = ['if', 'for', 'while', 'return', 'malloc', 'free', 'strcpy', 'strcat', 'sprintf', 'gets']
    for keyword in keywords:
        features.append(func_code.lower().count(keyword) / 10.0)
    
    # 符号特征
    symbols = ['{', '}', '(', ')', '[', ']', ';', '*', '&', '->']
    for symbol in symbols:
        features.append(func_code.count(symbol) / 20.0)
    
    # 填充到768维
    while len(features) < 768:
        # 使用代码哈希生成伪随机特征
        hash_val = hash(func_code + str(len(features))) % 10000
        features.append(hash_val / 10000.0)
    
    return features[:768]

def create_simple_graph(func_code, code_features):
    """创建简化的图结构"""
    
    # 基于代码行创建节点
    lines = func_code.split('\\n')
    max_nodes = min(50, max(5, len(lines)))  # 5-50个节点
    
    node_features = []
    edges = []
    
    # 创建节点特征
    for i in range(max_nodes):
        if i < len(lines):
            line = lines[i].strip()
            node_feature = create_line_features(line, code_features)
        else:
            node_feature = [0.0] * 768
        
        node_features.append(node_feature)
    
    # 创建边（简单的顺序连接 + 一些随机连接）
    for i in range(max_nodes - 1):
        # 顺序连接
        edges.append([i, 'AST', i + 1])
        
        # 添加一些控制流边
        if i % 3 == 0 and i + 2 < max_nodes:
            edges.append([i, 'CFG', i + 2])
        
        # 添加一些数据流边
        if i % 5 == 0 and i + 3 < max_nodes:
            edges.append([i, 'DFG', i + 3])
    
    # 创建序列特征
    sequence = []
    for i in range(6):  # 6个时间步
        start_idx = i * 128
        end_idx = min(start_idx + 128, 768)
        seq_step = code_features[start_idx:end_idx]
        
        if len(seq_step) < 128:
            seq_step.extend([0.0] * (128 - len(seq_step)))
        
        sequence.append(seq_step)
    
    return {
        'node_features': node_features,
        'edges': edges,
        'sequence': sequence
    }

def create_line_features(line, base_features):
    """为代码行创建特征"""
    
    features = base_features.copy()
    
    # 行特定特征
    features[0] = len(line) / 100.0
    features[1] = line.count(' ') / 20.0
    features[2] = 1.0 if '{' in line else 0.0
    features[3] = 1.0 if '}' in line else 0.0
    features[4] = 1.0 if 'if' in line.lower() else 0.0
    
    # 添加一些噪声以增加多样性
    for i in range(5, min(20, len(features))):
        noise = (hash(line + str(i)) % 1000) / 1000.0 * 0.1
        features[i] += noise
    
    return features

def main():
    """主函数"""
    
    print("🎯 从原始数据创建多分类数据集")
    print("基于CWE ID的漏洞类型分类")
    print()
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 创建多分类数据集
    df_filtered, mapping_info = create_multiclass_dataset_direct()
    
    # 2. 创建简化的训练数据
    output_dir = create_simplified_multiclass_data(df_filtered, mapping_info)
    
    print(f"\n🎉 多分类数据集创建完成!")
    print(f"📁 输出目录: {output_dir}")
    print(f"🏷️ 类别数量: {mapping_info['num_classes']}")
    print(f"📊 总样本数: {mapping_info['total_samples']}")
    
    # 显示类别分布
    print(f"\n📊 最终类别分布:")
    for i, cwe in mapping_info['label_to_cwe'].items():
        count = mapping_info['class_distribution'][str(i)]
        percentage = count / mapping_info['total_samples'] * 100
        print(f"   标签 {i}: {cwe} - {count} 样本 ({percentage:.1f}%)")
    
    return mapping_info

if __name__ == "__main__":
    main()
