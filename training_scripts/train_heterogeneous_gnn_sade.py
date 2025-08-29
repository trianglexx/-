#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构GNN + SADE混合模型训练脚本
结合异构图神经网络的强大特征提取能力与SADE自适应损失函数
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
from typing import Dict, List, Optional

# PyG Imports
try:
    from torch_geometric.data import Dataset, Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
    from torch_geometric.utils import add_self_loops, degree
except ImportError:
    print("❌ PyTorch Geometric不可用，请安装 pip install torch-geometric")
    exit(1)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] INFO: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class SADELoss(nn.Module):
    """
    SADE (Self-Adaptive Differential Evolution) 损失函数
    结合了交叉熵损失和自适应权重调整，专门为异构GNN优化
    """
    
    def __init__(self, num_classes, alpha=1.0, beta=2.0, gamma=0.5):
        super(SADELoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 基础权重
        self.beta = beta    # 类别平衡权重
        self.gamma = gamma  # 自适应调整权重
        
        # 初始化类别权重
        self.register_buffer('class_weights', torch.ones(num_classes))
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def update_class_statistics(self, targets):
        """更新类别统计信息"""
        with torch.no_grad():
            for i in range(self.num_classes):
                count = (targets == i).sum().float()
                self.class_counts[i] += count
                self.total_samples += count
            
            # 更新类别权重（逆频率权重）
            for i in range(self.num_classes):
                if self.class_counts[i] > 0:
                    freq = self.class_counts[i] / self.total_samples
                    self.class_weights[i] = 1.0 / (freq + 1e-8)
            
            # 归一化权重
            self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes
    
    def compute_sade_weights(self, predictions, targets):
        """计算SADE自适应权重"""
        batch_size = predictions.size(0)
        
        # 计算预测置信度
        probs = torch.softmax(predictions, dim=1)
        max_probs, pred_classes = torch.max(probs, dim=1)
        
        # 计算预测正确性
        correct_mask = (pred_classes == targets).float()
        
        # 自适应权重：错误预测的样本获得更高权重
        confidence_weights = 1.0 - max_probs
        correctness_weights = 1.0 - correct_mask
        
        # 结合置信度和正确性
        sade_weights = 1.0 + self.gamma * (confidence_weights + correctness_weights)
        
        return sade_weights
    
    def forward(self, predictions, targets):
        """前向传播计算损失"""
        # 更新类别统计
        self.update_class_statistics(targets)
        
        # 基础交叉熵损失
        ce_losses = self.ce_loss(predictions, targets)
        
        # 类别权重
        class_weights_batch = self.class_weights[targets]
        
        # SADE自适应权重
        sade_weights = self.compute_sade_weights(predictions, targets)
        
        # 组合损失
        total_loss = self.alpha * ce_losses + self.beta * ce_losses * class_weights_batch + sade_weights * ce_losses
        
        return total_loss.mean()

class HeterogeneousGNNLayer(MessagePassing):
    """异构图神经网络层 - 针对SADE损失函数优化"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_edge_types: int = 4,
                 alpha: float = 0.1,
                 dropout: float = 0.2,
                 aggr: str = 'add'):
        super(HeterogeneousGNNLayer, self).__init__(aggr=aggr)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_edge_types = num_edge_types
        self.alpha = alpha
        
        # 为每种边类型定义专用权重矩阵
        self.edge_type_weights = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(num_edge_types)
        ])
        
        # 自连接权重矩阵
        self.self_weight = nn.Linear(input_dim, output_dim, bias=True)
        
        # 输入投影（用于残差连接）
        if input_dim != output_dim:
            self.input_projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.input_projection = nn.Identity()
        
        # 边类型注意力权重
        self.edge_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * 2, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, 1),
                nn.LeakyReLU(0.2)
            ) for _ in range(num_edge_types)
        ])
        
        # 正则化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 层级别的边类型重要性权重
        self.layer_edge_importance = nn.Parameter(torch.ones(num_edge_types))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """参数初始化"""
        gain = nn.init.calculate_gain('relu')
        
        for i in range(self.num_edge_types):
            nn.init.xavier_uniform_(self.edge_type_weights[i].weight, gain=gain)
        
        nn.init.xavier_uniform_(self.self_weight.weight, gain=gain)
        
        if hasattr(self.input_projection, 'weight'):
            nn.init.xavier_uniform_(self.input_projection.weight, gain=gain)
    
    def forward(self, x, edge_index_dict, batch_size=None):
        """前向传播"""
        h_initial = x  # 保存初始特征用于残差连接
        h_projected = self.input_projection(h_initial)
        
        # 自连接变换
        h_self = self.self_weight(x)
        
        # 动态边权重计算
        edge_weights = F.softmax(self.layer_edge_importance, dim=0)
        
        # 聚合不同边类型的消息
        h_neighbors = []
        
        for edge_type_idx, (edge_type, edge_index) in enumerate(edge_index_dict.items()):
            if edge_index.size(1) == 0:  # 跳过空边
                continue
                
            # 使用对应边类型的权重矩阵
            edge_weight = self.edge_type_weights[edge_type_idx]
            
            # 消息传递
            h_edge = self.propagate(
                edge_index, 
                x=edge_weight(x), 
                edge_type_idx=edge_type_idx,
                size=None
            )
            
            # 应用边类型重要性权重
            h_edge = h_edge * edge_weights[edge_type_idx]
            h_neighbors.append(h_edge)
        
        # 聚合所有边类型的消息
        if h_neighbors:
            h_agg = torch.stack(h_neighbors, dim=0).sum(dim=0)
        else:
            h_agg = torch.zeros_like(h_self)
        
        # 组合自连接和邻居聚合
        h_out = (1 - self.alpha) * h_agg + h_self + self.alpha * h_projected
        
        # 正则化
        h_out = self.dropout(h_out)
        h_out = self.layer_norm(h_out)
        
        return h_out
    
    def message(self, x_j, edge_type_idx):
        """消息函数"""
        return x_j

class HeterogeneousGNN(nn.Module):
    """异构图神经网络模型 - 集成SADE损失优化"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 num_classes: int = 14,
                 num_layers: int = 3,
                 num_edge_types: int = 4,
                 dropout: float = 0.2,
                 alpha: float = 0.1):
        super(HeterogeneousGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 异构GNN层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                HeterogeneousGNNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_edge_types=num_edge_types,
                    alpha=alpha,
                    dropout=dropout
                )
            )
        
        # 序列投影层（将768维投影到hidden_dim）
        self.sequence_projection = nn.Linear(input_dim, hidden_dim)
        
        # 序列建模层（LSTM）
        self.sequence_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类层
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        # 全局边类型重要性（跨层共享）
        self.global_edge_importance = nn.Parameter(torch.ones(num_edge_types))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        """前向传播"""
        x, edge_index_dict, batch = data.x, data.edge_index_dict, data.batch
        sequence_features = data.sequence_features
        
        # 输入投影
        x = self.input_projection(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 保存初始特征
        x_initial = x
        
        # 异构GNN层
        for gnn_layer in self.gnn_layers:
            x_new = gnn_layer(x, edge_index_dict)
            x = x_new + x  # 残差连接
        
        # 图级别池化
        graph_features = global_mean_pool(x, batch)
        
        # 序列建模
        # 使用原始序列特征而不是构造虚拟序列
        batch_size = int(batch.max().item() + 1)
        
        # 重新组织序列特征为批次格式
        sequence_input_list = []
        for i in range(batch_size):
            # 从batch中找到对应样本的序列特征
            mask = (batch == i)
            sample_idx = mask.nonzero(as_tuple=True)[0][0]  # 获取第一个匹配的索引
            
            # 获取对应样本的序列特征
            seq_feat = sequence_features[sample_idx]  # [seq_len, 768]
            sequence_input_list.append(seq_feat)
        
        # 填充到相同长度
        max_seq_len = max(seq.size(0) for seq in sequence_input_list)
        padded_sequences = []
        
        for seq in sequence_input_list:
            if seq.size(0) < max_seq_len:
                padding = torch.zeros(max_seq_len - seq.size(0), seq.size(1)).to(seq.device)
                seq = torch.cat([seq, padding], dim=0)
            padded_sequences.append(seq)
        
        sequence_input = torch.stack(padded_sequences, dim=0)  # [batch_size, seq_len, 768]
        
        # 投影序列特征到hidden_dim
        sequence_input = self.sequence_projection(sequence_input)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM序列建模
        lstm_out, (h_n, c_n) = self.sequence_lstm(sequence_input)
        
        # 调试信息
        # print(f"LSTM input shape: {sequence_input.shape}")
        # print(f"LSTM output shape: {lstm_out.shape}")
        
        # 检查LSTM输出形状并正确处理
        if lstm_out.dim() == 3:  # [batch_size, seq_len, hidden_dim]
            sequence_features_final = lstm_out[:, -1, :]  # 取最后一个时间步
            
            # 注意力机制
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            attn_features = attn_out.mean(dim=1)  # 全局平均
        else:  # 其他情况处理
            sequence_features_final = lstm_out.mean(dim=0) if lstm_out.dim() > 1 else lstm_out
            attn_features = sequence_features_final
        
        # 确保特征维度匹配
        if sequence_features_final.dim() == 1:
            sequence_features_final = sequence_features_final.unsqueeze(0)
        if graph_features.dim() != sequence_features_final.dim():
            if graph_features.dim() == 2 and sequence_features_final.dim() == 1:
                sequence_features_final = sequence_features_final.unsqueeze(0).expand(graph_features.size(0), -1)
            elif graph_features.dim() == 1 and sequence_features_final.dim() == 2:
                graph_features = graph_features.unsqueeze(0).expand(sequence_features_final.size(0), -1)
        
        # 特征融合
        combined_features = torch.cat([graph_features, sequence_features_final], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_edge_importance_weights(self):
        """获取学习到的边类型重要性权重"""
        with torch.no_grad():
            weights = F.softmax(self.global_edge_importance, dim=0).cpu().numpy()
        return weights.tolist()

class VulnerabilityDataset(Dataset):
    """漏洞检测数据集"""
    
    def __init__(self, data_list):
        super(VulnerabilityDataset, self).__init__()
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]

def load_data(data_path: str):
    """加载LIVABLE格式数据并转换为PyG格式"""
    logger.info(f"📂 加载数据: {data_path}")
    
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    data_list = []
    edge_type_mapping = {'AST': 0, 'CFG': 1, 'DFG': 2, 'CDG': 3}
    
    for sample in tqdm(raw_data, desc="转换数据格式"):
        # 节点特征 (LIVABLE格式中是 'features')
        # 数据格式是 [seq_len, feature_dim]，我们取最后一个时间步作为节点特征
        features_seq = torch.tensor(sample['features'], dtype=torch.float32)
        
        # 对于图神经网络，我们需要将序列数据转换为单个特征向量
        # 方法1: 取最后一个时间步
        node_features = features_seq[-1:, :]  # [1, 768]
        
        # 方法2: 取平均 (可选)
        # node_features = features_seq.mean(dim=0, keepdim=True)  # [1, 768]
        
        num_nodes = node_features.shape[0]
        
        # 初始化边索引字典
        edge_index_dict = {}
        for edge_type_name in edge_type_mapping.keys():
            edge_index_dict[edge_type_name] = torch.tensor([], dtype=torch.long).view(2, 0)
        
        # 处理边 (LIVABLE格式中是 'structure')
        if 'structure' in sample:
            for edge_info in sample['structure']:
                source, edge_type, target = edge_info
                
                # 确保边的节点索引有效
                if source < num_nodes and target < num_nodes:
                    if edge_type in edge_type_mapping:
                        if edge_index_dict[edge_type].size(1) == 0:
                            edge_index_dict[edge_type] = torch.tensor([[source], [target]], dtype=torch.long)
                        else:
                            new_edge = torch.tensor([[source], [target]], dtype=torch.long)
                            edge_index_dict[edge_type] = torch.cat([edge_index_dict[edge_type], new_edge], dim=1)
        
        # 标签
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        # 创建Data对象，包含序列特征
        data = Data(
            x=node_features,  # 图节点特征 [1, 768]
            edge_index_dict=edge_index_dict,
            sequence_features=features_seq,  # 序列特征 [seq_len, 768]
            y=label
        )
        
        data_list.append(data)
    
    logger.info(f"✅ 成功加载 {len(data_list)} 个样本")
    return data_list

def train_model():
    """训练混合模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 使用设备: {device}")
    
    # 加载数据
    train_data = load_data('livable_multiclass_data/livable_train.json')
    valid_data = load_data('livable_multiclass_data/livable_valid.json')
    test_data = load_data('livable_multiclass_data/livable_test.json')
    
    # 创建数据集和数据加载器
    train_dataset = VulnerabilityDataset(train_data)
    valid_dataset = VulnerabilityDataset(valid_data)
    test_dataset = VulnerabilityDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = HeterogeneousGNN(
        input_dim=768,
        hidden_dim=256,
        num_classes=14,
        num_layers=3,
        num_edge_types=4,
        dropout=0.2,
        alpha=0.1
    ).to(device)
    
    # SADE损失函数
    criterion = SADELoss(num_classes=14, alpha=1.0, beta=2.0, gamma=0.5).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    
    # 训练参数
    num_epochs = 50
    best_valid_f1 = 0.0
    best_model = None
    patience = 10
    patience_counter = 0
    
    logger.info("🚀 开始训练混合模型 (异构GNN + SADE)")
    
    training_history = {
        'train_loss': [],
        'train_f1': [],
        'valid_f1': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 记录预测结果
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        _, _, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        # 验证阶段
        model.eval()
        valid_preds = []
        valid_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                outputs = model(batch)
                preds = torch.argmax(outputs, dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(batch.y.cpu().numpy())
        
        _, _, valid_f1, _ = precision_recall_fscore_support(valid_labels, valid_preds, average='weighted')
        
        # 记录历史
        training_history['train_loss'].append(train_loss)
        training_history['train_f1'].append(train_f1)
        training_history['valid_f1'].append(valid_f1)
        
        logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train F1={train_f1:.4f}, Valid F1={valid_f1:.4f}")
        
        # 保存最佳模型
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model = copy.deepcopy(model)
            patience_counter = 0
            logger.info(f"🎯 新的最佳验证F1: {best_valid_f1:.4f}")
        else:
            patience_counter += 1
        
        # 学习率调整
        scheduler.step(valid_f1)
        
        # 早停
        if patience_counter >= patience:
            logger.info(f"⏰ 早停触发，最佳验证F1: {best_valid_f1:.4f}")
            break
    
    # 测试阶段
    logger.info("🧪 开始测试...")
    best_model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = best_model(batch)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())
    
    # 计算测试指标
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
    
    # 获取边类型重要性
    edge_importance = best_model.get_edge_importance_weights()
    
    # 保存结果
    results = {
        'model_type': 'Heterogeneous_GNN_SADE',
        'best_valid_f1': best_valid_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision, 
        'test_recall': test_recall,
        'test_f1': test_f1,
        'edge_type_importance': edge_importance,
        'model_parameters': sum(p.numel() for p in best_model.parameters()),
        'sade_parameters': {
            'alpha': 1.0,
            'beta': 2.0, 
            'gamma': 0.5
        },
        'classification_report': classification_report(test_labels, test_preds, target_names=[f'CWE-{i}' for i in [119, 20, 399, 125, 264, 200, 189, 416, 190, 362, 476, 787, 284, 254]]),
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
    }
    
    # 创建结果目录
    results_dir = Path('heterogeneous_gnn_sade_results')
    results_dir.mkdir(exist_ok=True)
    
    # 保存结果
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存模型
    torch.save(best_model.state_dict(), results_dir / 'best_model.pth')
    
    # 保存训练历史
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("🎉 混合模型训练完成!")
    logger.info(f"📊 测试结果:")
    logger.info(f"   - 准确率: {test_accuracy:.4f}")
    logger.info(f"   - F1分数: {test_f1:.4f}")
    logger.info(f"   - 验证最佳F1: {best_valid_f1:.4f}")
    logger.info(f"🔗 边类型重要性:")
    edge_names = ['AST', 'CFG', 'DFG', 'CDG']
    for name, weight in zip(edge_names, edge_importance):
        logger.info(f"   - {name}: {weight:.4f}")

if __name__ == "__main__":
    train_model()