#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版异构GNN + SADE混合模型
基于现有异构GNN架构，替换损失函数为SADE
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import logging
from pathlib import Path
from tqdm import tqdm
import copy

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] INFO: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class SADELoss(nn.Module):
    """SADE自适应损失函数"""
    
    def __init__(self, num_classes, alpha=1.0, beta=2.0, gamma=0.5):
        super(SADELoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.register_buffer('class_weights', torch.ones(num_classes))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def update_class_statistics(self, targets):
        """更新类别统计信息"""
        with torch.no_grad():
            for i in range(self.num_classes):
                count = (targets == i).sum().float()
                self.class_counts[i] += count
                self.total_samples += count
            
            # 更新类别权重
            for i in range(self.num_classes):
                if self.class_counts[i] > 0:
                    freq = self.class_counts[i] / self.total_samples
                    self.class_weights[i] = 1.0 / (freq + 1e-8)
            
            self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes
    
    def compute_sade_weights(self, predictions, targets):
        """计算SADE自适应权重"""
        probs = torch.softmax(predictions, dim=1)
        max_probs, pred_classes = torch.max(probs, dim=1)
        
        correct_mask = (pred_classes == targets).float()
        
        confidence_weights = 1.0 - max_probs
        correctness_weights = 1.0 - correct_mask
        
        sade_weights = 1.0 + self.gamma * (confidence_weights + correctness_weights)
        
        return sade_weights
    
    def forward(self, predictions, targets):
        """前向传播计算损失"""
        self.update_class_statistics(targets)
        
        ce_losses = self.ce_loss(predictions, targets)
        class_weights_batch = self.class_weights[targets]
        sade_weights = self.compute_sade_weights(predictions, targets)
        
        total_loss = self.alpha * ce_losses + self.beta * ce_losses * class_weights_batch + sade_weights * ce_losses
        
        return total_loss.mean()

# 导入现有的异构GNN组件，但不执行其训练代码
import sys

# 只导入需要的类，不执行训练代码
with open('heterogeneous_gnn_pyg.py', 'r') as f:
    source_code = f.read()
    
# 只执行类定义和函数定义，跳过 if __name__ == '__main__' 部分
lines = source_code.split('\n')
filtered_lines = []
skip_main = False
for line in lines:
    if line.strip().startswith("if __name__ == '__main__':"):
        skip_main = True
        continue
    if not skip_main:
        filtered_lines.append(line)

filtered_code = '\n'.join(filtered_lines)
exec(filtered_code)

def train_heterogeneous_gnn_with_sade():
    """使用SADE损失函数训练异构GNN"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 使用设备: {device}")
    
    # 加载数据 - 只使用训练集进行学习分析
    logger.info("📂 加载训练数据...")
    with open('livable_multiclass_data/livable_train.json', 'r') as f:
        train_data = json.load(f)
    
    # 创建数据集和数据加载器 - 专注训练集
    train_dataset = HeterogeneousPygDataset(root=None, data_list=train_data, max_seq_len=6)
    
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 初始化模型
    model = HeterogeneousLIVABLEPygModel(
        input_dim=768,
        hidden_dim=256,
        num_classes=14,
        num_gnn_layers=3,
        num_edge_types=4,
        dropout=0.2,
        alpha=0.1
    ).to(device)
    
    # 使用SADE损失函数
    criterion = SADELoss(num_classes=14, alpha=1.0, beta=2.0, gamma=0.5).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    
    # 训练参数 - 专注训练集学习分析
    num_epochs = 50
    best_train_f1 = 0.0
    best_model = None
    
    logger.info("🚀 开始异构GNN + SADE训练集学习分析 (50轮)")
    
    training_history = {
        'train_loss': [],
        'train_f1': [],
        'train_accuracy': [],
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
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        _, _, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
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
        training_history['train_f1'].append(train_f1)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['class_accuracies_history'].append(class_accuracies.copy())
        
        logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_accuracy:.4f}, Train F1={train_f1:.4f}")
        
        # 保存最佳训练模型
        if train_f1 > best_train_f1:
            best_train_f1 = train_f1
            best_model = copy.deepcopy(model)
            logger.info(f"🎯 新的最佳训练F1: {best_train_f1:.4f}")
        
        # 学习率调整（基于训练F1）
        scheduler.step(train_f1)
    
    # 最终训练集分析
    logger.info("📊 分析训练集学习效果...")
    
    # 获取边类型重要性
    edge_importance = best_model.get_edge_type_importance().cpu().detach().numpy().tolist()
    
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
    final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    final_classification_report = classification_report(all_labels, all_preds, target_names=cwe_names, output_dict=True)
    
    logger.info(f"{'CWE类别':<12} | {'准确率':<10} | {'精确率':<10} | {'召回率':<10} | {'F1分数':<10}")
    logger.info("-" * 70)
    
    for i, cwe in enumerate(cwe_names):
        accuracy = last_epoch[cwe]
        precision = final_precision[i] if i < len(final_precision) else 0.0
        recall = final_recall[i] if i < len(final_recall) else 0.0
        f1 = final_f1[i] if i < len(final_f1) else 0.0
        
        logger.info(f"{cwe:<12} | {accuracy:>9.4f} | {precision:>9.4f} | {recall:>9.4f} | {f1:>9.4f}")
    
    # 保存详细结果
    results = {
        'model_type': 'Heterogeneous_GNN_SADE_Training_Analysis',
        'final_train_f1': best_train_f1,
        'final_train_accuracy': training_history['train_accuracy'][-1],
        'final_train_loss': training_history['train_loss'][-1],
        'edge_type_importance': edge_importance,
        'model_parameters': sum(p.numel() for p in best_model.parameters()),
        'sade_parameters': {
            'alpha': 1.0,
            'beta': 2.0,
            'gamma': 0.5
        },
        'training_history': training_history,
        'class_accuracy_analysis': {
            'initial': first_epoch,
            'final': last_epoch,
            'improvements': {cwe: last_epoch[cwe] - first_epoch[cwe] for cwe in cwe_names}
        },
        'final_classification_report': final_classification_report
    }
    
    # 创建结果目录
    results_dir = Path('heterogeneous_gnn_sade_training_analysis')
    results_dir.mkdir(exist_ok=True)
    
    # 保存结果
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    torch.save(best_model.state_dict(), results_dir / 'best_model.pth')
    
    logger.info("🎉 异构GNN + SADE训练分析完成!")
    logger.info(f"📊 最终训练结果:")
    logger.info(f"   - 最佳训练F1分数: {best_train_f1:.4f}")
    logger.info(f"   - 最终训练准确率: {training_history['train_accuracy'][-1]:.4f}")
    logger.info(f"   - 最终训练损失: {training_history['train_loss'][-1]:.4f}")
    logger.info(f"🔗 学习到的边类型重要性:")
    edge_names = ['AST', 'CFG', 'DFG', 'CDG']
    for name, weight in zip(edge_names, edge_importance):
        logger.info(f"   - {name}: {weight:.4f}")
    
    # 找出改进最大和最小的类别
    improvements = {cwe: last_epoch[cwe] - first_epoch[cwe] for cwe in cwe_names}
    best_improved = max(improvements, key=improvements.get)
    worst_improved = min(improvements, key=improvements.get)
    
    logger.info(f"🏆 改进最大的类别: {best_improved} (+{improvements[best_improved]:.4f})")
    logger.info(f"📉 改进最小的类别: {worst_improved} ({improvements[worst_improved]:+.4f})")

if __name__ == "__main__":
    train_heterogeneous_gnn_with_sade()