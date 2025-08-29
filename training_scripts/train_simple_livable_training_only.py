#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版LIVABLE多分类训练脚本 - 训练集学习分析版本
只在训练集上学习50轮，观察模型学习效果
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from datetime import datetime
import logging
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import copy

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimpleLIVABLEModel(nn.Module):
    """简化版LIVABLE模型，基于图神经网络和序列模型"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=14, seq_input_dim=128):
        super(SimpleLIVABLEModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.seq_input_dim = seq_input_dim
        
        # 图分支 - 使用简单的MLP处理节点特征
        self.graph_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 序列分支 - 使用GRU处理序列
        self.seq_encoder = nn.GRU(
            input_size=seq_input_dim,  # 序列特征维度
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim),  # 图特征 + 双向GRU特征
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, node_features, sequences):
        """前向传播"""
        batch_size = node_features.size(0)
        
        # 图分支：处理节点特征
        # node_features: [batch_size, num_nodes, input_dim]
        graph_features = self.graph_encoder(node_features)  # [batch_size, num_nodes, hidden_dim]
        
        # 图级别池化
        graph_pooled = torch.mean(graph_features, dim=1)  # [batch_size, hidden_dim]
        
        # 序列分支：处理序列特征
        # sequences: [batch_size, seq_len, seq_dim]
        seq_output, _ = self.seq_encoder(sequences)  # [batch_size, seq_len, hidden_dim*2]
        
        # 序列级别池化
        seq_pooled = torch.mean(seq_output, dim=1)  # [batch_size, hidden_dim*2]
        
        # 特征融合
        combined = torch.cat([graph_pooled, seq_pooled], dim=1)  # [batch_size, hidden_dim + hidden_dim*2]
        fused_features = self.fusion(combined)  # [batch_size, hidden_dim]
        
        # 分类
        logits = self.classifier(fused_features)  # [batch_size, num_classes]
        
        return logits

class SimpleLIVABLEDataset(Dataset):
    """简化版数据集"""
    def __init__(self, data_path, max_nodes=100, max_seq_len=512):
        self.data = []
        self.max_nodes = max_nodes
        self.max_seq_len = max_seq_len
        
        logger.info(f"📥 加载数据: {data_path}")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        logger.info(f"📊 处理 {len(raw_data)} 个样本...")
        for item in tqdm(raw_data, desc="处理数据"):
            try:
                features = item['features']  # 节点特征
                target = item['label'][0][0]  # 标签
                sequence = item['sequence']  # 序列
                
                # 处理节点特征 - 填充或截断到固定大小
                if len(features) > self.max_nodes:
                    features = features[:self.max_nodes]
                else:
                    # 用零填充
                    padding = [[0.0] * len(features[0])] * (self.max_nodes - len(features))
                    features = features + padding
                
                # 处理序列 - 序列已经是特征向量的列表
                seq_features = sequence  # 直接使用原始序列特征

                # 确保序列特征长度
                target_seq_len = 64  # 目标序列长度
                if len(seq_features) > target_seq_len:
                    seq_features = seq_features[:target_seq_len]
                else:
                    # 用零向量填充
                    if len(seq_features) > 0:
                        feat_dim = len(seq_features[0])  # 获取特征维度
                        padding = [[0.0] * feat_dim] * (target_seq_len - len(seq_features))
                        seq_features = seq_features + padding
                    else:
                        # 如果序列为空，创建零向量
                        seq_features = [[0.0] * 128] * target_seq_len
                
                self.data.append({
                    'features': torch.FloatTensor(features),
                    'sequence': torch.FloatTensor(seq_features),
                    'target': target
                })
                
            except Exception as e:
                logger.warning(f"跳过无效样本: {e}")
                continue
        
        logger.info(f"✅ 成功加载 {len(self.data)} 个样本")

        # 获取序列特征维度
        if len(self.data) > 0:
            self.seq_feature_dim = len(self.data[0]['sequence'][0])
            logger.info(f"📏 序列特征维度: {self.seq_feature_dim}")
        else:
            self.seq_feature_dim = 128
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """批处理函数"""
    features = torch.stack([item['features'] for item in batch])
    sequences = torch.stack([item['sequence'] for item in batch])
    targets = torch.LongTensor([item['target'] for item in batch])
    
    return features, sequences, targets

def train_simple_livable_training_analysis():
    """训练简化版LIVABLE - 专注训练集学习分析"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 使用设备: {device}")
    
    # 加载训练数据
    logger.info("📂 加载训练数据...")
    train_dataset = SimpleLIVABLEDataset('livable_multiclass_data/livable_train.json')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建模型
    logger.info("🏗️ 创建简化版LIVABLE模型...")
    seq_feature_dim = train_dataset.seq_feature_dim
    model = SimpleLIVABLEModel(
        input_dim=768,
        hidden_dim=256,
        num_classes=14,
        seq_input_dim=seq_feature_dim
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练参数 - 专注训练集学习分析
    num_epochs = 50
    best_train_f1 = 0.0
    best_model = None
    
    logger.info("🚀 开始简化LIVABLE训练集学习分析 (50轮)")
    
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1': [],
        'class_accuracies_history': []
    }
    
    # CWE类别名称
    cwe_names = ['CWE-119', 'CWE-20', 'CWE-399', 'CWE-125', 'CWE-264', 'CWE-200', 
                 'CWE-189', 'CWE-416', 'CWE-190', 'CWE-362', 'CWE-476', 'CWE-787', 
                 'CWE-284', 'CWE-254']
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for features, sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = features.to(device)
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features, sequences)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
        
        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        _, _, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for i in range(14):
            mask = np.array(all_labels) == i
            if mask.sum() > 0:
                class_acc = (np.array(all_preds)[mask] == i).sum() / mask.sum()
                class_accuracies[cwe_names[i]] = class_acc
            else:
                class_accuracies[cwe_names[i]] = 0.0
        
        # 记录历史
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['train_f1'].append(train_f1)
        training_history['class_accuracies_history'].append(class_accuracies.copy())
        
        logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, Train F1={train_f1:.4f}")
        
        # 保存最佳训练模型
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            best_model = copy.deepcopy(model)
            logger.info(f"🎯 新的最佳训练F1: {best_train_f1:.4f}")
    
    # 最终训练集分析
    logger.info("📊 分析训练集学习效果...")
    
    # 分析类别准确率变化
    logger.info("🎯 各类别训练准确率变化分析:")
    logger.info("=" * 80)
    
    first_epoch = training_history['class_accuracies_history'][0]
    last_epoch = training_history['class_accuracies_history'][-1]
    
    logger.info(f"{'CWE类别':<12} | {'初始准确率':<12} | {'最终准确率':<12} | {'提升幅度':<12} | {'相对提升':<12}")
    logger.info("-" * 80)
    
    for cwe in cwe_names:
        initial_acc = first_epoch[cwe]
        final_acc = last_epoch[cwe]
        improvement = final_acc - initial_acc
        relative_improvement = (improvement / (initial_acc + 1e-8)) * 100 if initial_acc > 0 else float('inf') if improvement > 0 else 0
        
        logger.info(f"{cwe:<12} | {initial_acc:>11.4f} | {final_acc:>11.4f} | {improvement:>+11.4f} | {relative_improvement:>+10.1f}%")
    
    # 计算最终训练集各类别详细指标
    logger.info("\n📋 最终训练集各类别详细指标:")
    logger.info("=" * 80)
    
    # 使用最后一轮的预测结果
    final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    final_classification_report = classification_report(all_labels, all_preds, target_names=cwe_names, output_dict=True, zero_division=0)
    
    logger.info(f"{'CWE类别':<12} | {'准确率':<10} | {'精确率':<10} | {'召回率':<10} | {'F1分数':<10}")
    logger.info("-" * 70)
    
    for i, cwe in enumerate(cwe_names):
        accuracy = last_epoch[cwe]
        precision = final_precision[i] if i < len(final_precision) else 0.0
        recall = final_recall[i] if i < len(final_recall) else 0.0
        f1 = final_f1[i] if i < len(final_f1) else 0.0
        
        logger.info(f"{cwe:<12} | {accuracy:>9.4f} | {precision:>9.4f} | {recall:>9.4f} | {f1:>9.4f}")
    
    # 保存结果
    results = {
        'model_type': 'Simple_LIVABLE_Training_Analysis',
        'final_train_f1': best_train_f1,
        'final_train_accuracy': training_history['train_accuracy'][-1],
        'final_train_loss': training_history['train_loss'][-1],
        'model_parameters': sum(p.numel() for p in best_model.parameters()),
        'training_history': training_history,
        'class_accuracy_analysis': {
            'initial': first_epoch,
            'final': last_epoch,
            'improvements': {cwe: last_epoch[cwe] - first_epoch[cwe] for cwe in cwe_names}
        },
        'final_classification_report': final_classification_report
    }
    
    # 创建结果目录
    results_dir = Path('simple_livable_training_analysis')
    results_dir.mkdir(exist_ok=True)
    
    # 保存结果
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    torch.save(best_model.state_dict(), results_dir / 'best_model.pth')
    
    logger.info("🎉 简化LIVABLE训练分析完成!")
    logger.info(f"📊 最终训练结果:")
    logger.info(f"   - 最佳训练F1分数: {best_train_f1:.4f}")
    logger.info(f"   - 最终训练准确率: {training_history['train_accuracy'][-1]:.4f}")
    logger.info(f"   - 最终训练损失: {training_history['train_loss'][-1]:.4f}")
    
    # 找出改进最大和最小的类别
    improvements = {cwe: last_epoch[cwe] - first_epoch[cwe] for cwe in cwe_names}
    best_improved = max(improvements, key=improvements.get)
    worst_improved = min(improvements, key=improvements.get)
    
    logger.info(f"🏆 改进最大的类别: {best_improved} (+{improvements[best_improved]:.4f})")
    logger.info(f"📉 改进最小的类别: {worst_improved} ({improvements[worst_improved]:+.4f})")

if __name__ == "__main__":
    train_simple_livable_training_analysis()