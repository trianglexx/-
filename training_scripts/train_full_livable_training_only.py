#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整版LIVABLE模型训练脚本 - 训练集学习分析版本
使用原始LIVABLE架构（APPNP + 双向GRU）只在训练集上学习50轮
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

# --- PyG Imports ---
try:
    from torch_geometric.data import Dataset, Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import APPNP, global_mean_pool
except ImportError:
    print("PyTorch Geometric is not installed. Please install it.")
    exit(1)

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] INFO: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Re-implemented MLPReadout for consistency ---
class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

# --- PyG Dataset ---
class FullLIVABLEPygDataset(Dataset):
    def __init__(self, root, data_list, max_seq_len=128):
        self.data_list = data_list
        self.max_seq_len = max_seq_len
        super(FullLIVABLEPygDataset, self).__init__(root)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        item = self.data_list[idx]
        
        features = torch.FloatTensor(item['features'])
        num_nodes = features.shape[0]

        edges = item['structure']
        sources, dests = [], []
        for s, _, t in edges:
            if s < num_nodes and t < num_nodes:
                sources.append(s)
                dests.append(t)
        edge_index = torch.LongTensor([sources, dests])

        sequence = torch.FloatTensor(item['sequence'])
        if sequence.shape[0] > self.max_seq_len:
            sequence = sequence[:self.max_seq_len, :]
        elif sequence.shape[0] < self.max_seq_len:
            pad_size = self.max_seq_len - sequence.shape[0]
            feat_dim = sequence.shape[1] if sequence.shape[0] > 0 else 128
            padding = torch.zeros(pad_size, feat_dim)
            sequence = torch.cat((sequence, padding), dim=0)
        
        label = torch.LongTensor([item['label'][0][0]])

        return Data(x=features, edge_index=edge_index, sequence=sequence, y=label)

# --- PyG Model ---
class FullLIVABLEPygModel(nn.Module):
    def __init__(self, input_dim=768, seq_input_dim=128, num_classes=14):
        super(FullLIVABLEPygModel, self).__init__()
        self.seq_hid = 512
        self.max_seq_len = 128

        # Graph Branch components - 原始LIVABLE的APPNP
        self.appnp = APPNP(K=16, alpha=0.1)  # 原始LIVABLE参数
        self.mlp_graph = MLPReadout(input_dim, num_classes)

        # Sequence Branch components - 原始LIVABLE的双向GRU
        self.bigru_seq = nn.GRU(seq_input_dim, self.seq_hid, num_layers=1, bidirectional=True, batch_first=True)
        self.mlp_seq = MLPReadout(2 * self.seq_hid, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, sequence, batch = data.x, data.edge_index, data.sequence, data.batch

        # --- Sequence Branch (原始LIVABLE) ---
        batch_size = data.num_graphs
        # Reshape sequence from (B * L, D) to (B, L, D)
        sequence = sequence.view(batch_size, self.max_seq_len, -1)
        seq_out, _ = self.bigru_seq(sequence)
        seq_out = torch.transpose(seq_out, 1, 2)
        seq1 = F.avg_pool1d(seq_out, seq_out.size(2)).squeeze(2)
        seq2 = F.max_pool1d(seq_out, seq_out.size(2)).squeeze(2)
        seq_outputs = self.mlp_seq(self.dropout(seq1 + seq2))

        # --- Graph Branch (原始LIVABLE APPNP) ---
        if x.numel() > 0:
            # 使用原始LIVABLE的APPNP算法
            x = self.appnp(x, edge_index)
            graph_pooled = global_mean_pool(x, batch)
            graph_outputs = self.mlp_graph(self.dropout(graph_pooled))
        else:
            graph_outputs = torch.zeros_like(seq_outputs)

        # 原始LIVABLE的相加融合策略
        return graph_outputs + seq_outputs

def train_full_livable_training_analysis():
    """训练完整版LIVABLE - 专注训练集学习分析"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 使用设备: {device}")
    
    # 加载训练数据
    logger.info("📂 加载训练数据...")
    with open('livable_multiclass_data/livable_train.json', 'r') as f:
        train_data = [item for item in json.load(f) if len(item['features']) > 0]
    
    # 创建数据集和数据加载器
    train_dataset = FullLIVABLEPygDataset(root=None, data_list=train_data)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 创建完整版LIVABLE模型
    model = FullLIVABLEPygModel().to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数 - 专注训练集学习分析
    num_epochs = 50
    best_train_f1 = 0.0
    best_model = None
    
    logger.info("🚀 开始完整LIVABLE训练集学习分析 (50轮)")
    
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
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, data.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
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
        'model_type': 'Full_LIVABLE_Training_Analysis',
        'final_train_f1': best_train_f1,
        'final_train_accuracy': training_history['train_accuracy'][-1],
        'final_train_loss': training_history['train_loss'][-1],
        'model_parameters': sum(p.numel() for p in best_model.parameters()),
        'livable_parameters': {
            'appnp_k': 16,
            'appnp_alpha': 0.1,
            'seq_hid': 512
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
    results_dir = Path('full_livable_training_analysis')
    results_dir.mkdir(exist_ok=True)
    
    # 保存结果
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    torch.save(best_model.state_dict(), results_dir / 'best_model.pth')
    
    logger.info("🎉 完整LIVABLE训练分析完成!")
    logger.info(f"📊 最终训练结果:")
    logger.info(f"   - 最佳训练F1分数: {best_train_f1:.4f}")
    logger.info(f"   - 最终训练准确率: {training_history['train_accuracy'][-1]:.4f}")
    logger.info(f"   - 最终训练损失: {training_history['train_loss'][-1]:.4f}")
    logger.info(f"🔧 原始LIVABLE架构参数:")
    logger.info(f"   - APPNP: K=16, α=0.1")
    logger.info(f"   - 双向GRU隐藏层: {512}")
    
    # 找出改进最大和最小的类别
    improvements = {cwe: last_epoch[cwe] - first_epoch[cwe] for cwe in cwe_names}
    best_improved = max(improvements, key=improvements.get)
    worst_improved = min(improvements, key=improvements.get)
    
    logger.info(f"🏆 改进最大的类别: {best_improved} (+{improvements[best_improved]:.4f})")
    logger.info(f"📉 改进最小的类别: {worst_improved} ({improvements[worst_improved]:+.4f})")

if __name__ == "__main__":
    train_full_livable_training_analysis()