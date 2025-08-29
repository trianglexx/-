#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构图神经网络模型 - PyTorch Geometric版本
基于异构边类型的改进LIVABLE架构，使用PyG实现
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


class HeterogeneousGNNLayer(MessagePassing):
    """
    异构图神经网络层 - PyG实现
    
    实现公式: h_i^{(l+1)} = σ((1-α) Σ_{r∈R} Σ_{j∈N_i^r} (1/c_{i,r}) W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)} + α h_i^{(0)})
    """
    
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
        
        # 为每种边类型定义专用权重矩阵 W_r
        self.edge_type_weights = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(num_edge_types)
        ])
        
        # 自连接权重矩阵 W_0
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
        
        # 边类型嵌入
        self.edge_type_embeddings = nn.Embedding(num_edge_types, output_dim // 4)
        
        # 层级别的边类型重要性权重
        self.layer_edge_importance = nn.Parameter(torch.ones(num_edge_types))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """参数初始化"""
        gain = nn.init.calculate_gain('relu')
        
        for i in range(self.num_edge_types):
            nn.init.xavier_uniform_(self.edge_type_weights[i].weight, gain=gain)
        
        nn.init.xavier_uniform_(self.self_weight.weight, gain=gain)
        if self.self_weight.bias is not None:
            nn.init.zeros_(self.self_weight.bias)
        
        if hasattr(self.input_projection, 'weight'):
            nn.init.xavier_uniform_(self.input_projection.weight, gain=gain)
        
        nn.init.uniform_(self.edge_type_embeddings.weight, -0.1, 0.1)
        
        # 基于漏洞检测理论初始化边类型重要性
        with torch.no_grad():
            prior_weights = torch.tensor([1.2, 1.1, 1.4, 0.9])  # AST, CFG, DFG, CDG
            self.layer_edge_importance.data = prior_weights
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: Optional[torch.Tensor] = None,
                initial_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E] 
            edge_type: 边类型 [E] (可选，默认为0)
            initial_x: 初始特征（用于残差连接）[N, output_dim]
            
        Returns:
            更新后的节点特征 [N, output_dim]
        """
        if initial_x is None:
            initial_x = self.input_projection(x)
        
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        
        # 计算边类型重要性权重
        edge_weights = F.softmax(self.layer_edge_importance, dim=0)
        
        # 为每种边类型分别进行消息传递
        type_messages = []
        
        for r in range(self.num_edge_types):
            # 筛选当前边类型的边
            edge_mask = (edge_type == r)
            
            if edge_mask.sum() > 0:
                # 获取该类型的边
                type_edge_index = edge_index[:, edge_mask]
                
                # 应用边类型特定的变换
                transformed_x = self.edge_type_weights[r](x)
                
                # 消息传递
                messages = self.propagate(
                    type_edge_index, 
                    x=transformed_x, 
                    original_x=x,
                    edge_type_id=r
                )
                
                # 应用边类型重要性权重
                weighted_messages = messages * edge_weights[r]
                type_messages.append(weighted_messages)
            else:
                # 如果没有该类型的边，添加零消息
                type_messages.append(torch.zeros(x.size(0), self.output_dim, device=x.device))
        
        # 聚合所有边类型的消息
        aggregated_messages = torch.stack(type_messages, dim=0).sum(dim=0)
        
        # 自连接
        self_transformed = self.self_weight(x)
        
        # 组合: (1-α) * 邻居消息 + 自连接 + α * 初始特征
        output = ((1 - self.alpha) * aggregated_messages + 
                 self_transformed + 
                 self.alpha * initial_x)
        
        # 应用dropout和层归一化
        output = self.dropout(output)
        output = self.layer_norm(output)
        output = F.gelu(output)
        
        return output
    
    def get_layer_edge_importance(self):
        """获取层级别的边类型重要性权重"""
        return F.softmax(self.layer_edge_importance, dim=0)
    
    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, 
                original_x_j: torch.Tensor, original_x_i: torch.Tensor,
                edge_type_id: int, index: torch.Tensor) -> torch.Tensor:
        """计算消息"""
        # 计算注意力权重
        concat_features = torch.cat([original_x_i, original_x_j], dim=1)
        attention_scores = self.edge_attention[edge_type_id](concat_features)
        
        # 边类型嵌入增强
        edge_emb = self.edge_type_embeddings(
            torch.full((attention_scores.size(0),), edge_type_id, 
                      dtype=torch.long, device=x_j.device)
        )
        enhanced_scores = attention_scores + torch.mean(edge_emb, dim=1, keepdim=True)
        
        # 软注意力
        alpha = F.softmax(enhanced_scores, dim=0)
        
        return alpha * x_j


class MLPReadout(nn.Module):
    """多层感知机读出层"""
    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        layers = []
        dim = input_dim
        
        for l in range(L):
            layers.append(nn.Linear(dim, dim // 2, bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            dim = dim // 2
        
        layers.append(nn.Linear(dim, output_dim, bias=True))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class HeterogeneousLIVABLEPygModel(nn.Module):
    """基于PyTorch Geometric的异构LIVABLE模型"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 seq_input_dim: int = 128,
                 num_classes: int = 14,
                 num_edge_types: int = 4,
                 num_gnn_layers: int = 3,
                 alpha: float = 0.15,
                 dropout: float = 0.2,
                 max_seq_len: int = 6):
        super(HeterogeneousLIVABLEPygModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.max_seq_len = max_seq_len
        self.seq_hid = 256
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 异构GNN层
        self.hetero_gnn_layers = nn.ModuleList([
            HeterogeneousGNNLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_edge_types=num_edge_types,
                alpha=alpha,
                dropout=dropout
            ) for _ in range(num_gnn_layers)
        ])
        
        # 图分支MLP
        self.graph_mlp = MLPReadout(hidden_dim, num_classes, L=2)
        
        # 序列分支 - 双向GRU
        self.sequence_gru = nn.GRU(
            seq_input_dim, 
            self.seq_hid, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )
        
        # 序列分支MLP
        self.sequence_mlp = MLPReadout(2 * self.seq_hid, num_classes, L=2)
        
        # 全局dropout
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的边类型重要性权重 - 关键改进
        self.edge_type_importance = nn.Parameter(torch.ones(num_edge_types))
        self._init_edge_importance()
        
    def forward(self, data):
        """前向传播"""
        x, edge_index, sequence, batch = data.x, data.edge_index, data.sequence, data.batch
        
        # 获取边类型（如果可用）
        edge_type = getattr(data, 'edge_type', None)
        
        # --- 序列分支 ---
        batch_size = data.num_graphs
        # 重塑序列: (B * L, D) -> (B, L, D)
        sequence = sequence.view(batch_size, self.max_seq_len, -1)
        seq_out, _ = self.sequence_gru(sequence)
        
        # 序列池化
        seq_out = torch.transpose(seq_out, 1, 2)
        seq1 = F.avg_pool1d(seq_out, seq_out.size(2)).squeeze(2)
        seq2 = F.max_pool1d(seq_out, seq_out.size(2)).squeeze(2)
        seq_outputs = self.sequence_mlp(self.dropout(seq1 + seq2))
        
        # --- 异构图分支 ---
        if x.numel() > 0:
            # 输入投影
            h = self.input_projection(x)
            initial_h = h.clone()
            
            # 通过异构GNN层
            for i, gnn_layer in enumerate(self.hetero_gnn_layers):
                h = gnn_layer(h, edge_index, edge_type, initial_h)
            
            # 图级池化
            graph_pooled = global_mean_pool(h, batch)
            graph_outputs = self.graph_mlp(self.dropout(graph_pooled))
        else:
            graph_outputs = torch.zeros_like(seq_outputs)
        
        # 组合输出
        return graph_outputs + seq_outputs
    
    def _init_edge_importance(self):
        """基于漏洞检测领域知识初始化边类型重要性"""
        with torch.no_grad():
            # AST: 语法结构, CFG: 控制流, DFG: 数据流, CDG: 控制依赖
            # 基于理论，DFG对漏洞检测最重要，AST次之
            prior_weights = torch.tensor([1.2, 1.1, 1.4, 0.9])
            self.edge_type_importance.data = prior_weights
    
    def get_edge_type_importance(self):
        """获取边类型重要性权重"""
        return F.softmax(self.edge_type_importance, dim=0)


class HeterogeneousPygDataset(Dataset):
    """异构图PyG数据集"""
    
    def __init__(self, root, data_list, max_seq_len=6):
        self.data_list = data_list
        self.max_seq_len = max_seq_len
        
        # 边类型映射
        self.edge_type_mapping = {
            'AST': 0,
            'CFG': 1,
            'DFG': 2,
            'CDG': 3
        }
        
        super(HeterogeneousPygDataset, self).__init__(root)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        item = self.data_list[idx]
        
        # 节点特征
        features = torch.FloatTensor(item['features'])
        num_nodes = features.shape[0]
        
        # 边和边类型
        edges = item['structure']
        sources, dests, edge_types = [], [], []
        
        for s, edge_type_str, t in edges:
            if s < num_nodes and t < num_nodes:
                sources.append(s)
                dests.append(t)
                # 映射边类型
                edge_type_id = self.edge_type_mapping.get(edge_type_str, 0)
                edge_types.append(edge_type_id)
        
        edge_index = torch.LongTensor([sources, dests])
        edge_type = torch.LongTensor(edge_types)
        
        # 序列特征
        sequence = torch.FloatTensor(item['sequence'])
        if sequence.shape[0] > self.max_seq_len:
            sequence = sequence[:self.max_seq_len, :]
        elif sequence.shape[0] < self.max_seq_len:
            pad_size = self.max_seq_len - sequence.shape[0]
            feat_dim = sequence.shape[1] if sequence.shape[0] > 0 else 128
            padding = torch.zeros(pad_size, feat_dim)
            sequence = torch.cat((sequence, padding), dim=0)
        
        # 标签
        label = torch.LongTensor([item['label'][0][0]])
        
        return Data(x=features, edge_index=edge_index, edge_type=edge_type, 
                   sequence=sequence, y=label)


def load_data_lists():
    """加载数据列表"""
    logger.info("📥 加载数据列表...")
    
    with open('livable_multiclass_data/livable_train.json', 'r') as f:
        train_data = [item for item in json.load(f) if len(item['features']) > 0]
    with open('livable_multiclass_data/livable_valid.json', 'r') as f:
        valid_data = [item for item in json.load(f) if len(item['features']) > 0]
    with open('livable_multiclass_data/livable_test.json', 'r') as f:
        test_data = [item for item in json.load(f) if len(item['features']) > 0]
    
    logger.info(f"✅ 加载完成: {len(train_data)} 训练, {len(valid_data)} 验证, {len(test_data)} 测试样本")
    return train_data, valid_data, test_data


def evaluate(model, data_loader, device, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data.y)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    _, _, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, 
                                                 average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1, all_predictions, all_targets


def main():
    """主训练函数"""
    print("🚀 异构图神经网络训练开始 (PyG版本)")
    print("基于边类型异构性的LIVABLE改进模型")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 使用设备: {device}")
    
    # 加载数据
    train_list, valid_list, test_list = load_data_lists()
    
    # 创建数据集
    train_dataset = HeterogeneousPygDataset(root='pyg_data/hetero_train', data_list=train_list)
    valid_dataset = HeterogeneousPygDataset(root='pyg_data/hetero_valid', data_list=valid_list)
    test_dataset = HeterogeneousPygDataset(root='pyg_data/hetero_test', data_list=test_list)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    logger.info(f"📊 数据加载器: 训练{len(train_loader)}, 验证{len(valid_loader)}, 测试{len(test_loader)}批次")
    
    # 创建模型
    model = HeterogeneousLIVABLEPygModel(
        input_dim=768,
        hidden_dim=256,
        seq_input_dim=128,
        num_classes=14,
        num_edge_types=4,
        num_gnn_layers=3,
        alpha=0.15,
        dropout=0.2
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 异构GNN模型: {param_count:,} 参数")
    
    # 设置优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练参数
    num_epochs, patience, best_valid_f1, epochs_no_improve = 50, 10, 0, 0
    best_model_state = None
    
    logger.info(f"🎯 开始训练，最多{num_epochs}轮，早停耐心值={patience}")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for data in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, data.y)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 验证阶段
        valid_loss, valid_acc, valid_f1, _, _ = evaluate(model, valid_loader, device, criterion)
        
        logger.info(f"Epoch {epoch+1} | 训练损失: {avg_train_loss:.4f} | "
                   f"验证损失: {valid_loss:.4f}, 准确率: {valid_acc:.4f}, F1: {valid_f1:.4f}")
        
        # 保存最佳模型
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"🏆 新的最佳验证F1: {best_valid_f1:.4f}")
        else:
            epochs_no_improve += 1
        
        # 早停检查
        if epochs_no_improve >= patience:
            logger.info(f"⚠️ 早停触发，在第{epoch+1}轮停止")
            break
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info("💾 已加载最佳模型进行最终评估")
    
    # 最终测试
    logger.info("📊 在测试集上评估最佳模型...")
    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate(model, test_loader, device, criterion)
    
    print("\n" + "=" * 60)
    print("🎉 异构GNN训练完成!")
    print(f"🏆 最佳验证F1: {best_valid_f1:.4f}")
    print(f"🎯 最终测试结果:")
    print(f"   准确率: {test_acc:.4f}")
    print(f"   F1分数: {test_f1:.4f}")
    
    # 边类型重要性分析
    edge_importance = model.get_edge_type_importance()
    edge_names = ['AST', 'CFG', 'DFG', 'CDG']
    print(f"\n🔗 学习到的边类型重要性:")
    for i, (name, importance) in enumerate(zip(edge_names, edge_importance)):
        print(f"   {name}: {importance.item():.4f}")
    
    # 详细分类报告
    with open('multiclass_label_mapping.json', 'r') as f:
        label_map = json.load(f)
    class_names = [label_map['label_to_cwe'][str(i)] for i in range(14)]
    
    print(f"\n📋 测试集分类报告:")
    report = classification_report(test_targets, test_preds, target_names=class_names, zero_division=0)
    print(report)
    
    # 保存结果
    results = {
        'best_valid_f1': best_valid_f1,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'edge_type_importance': edge_importance.cpu().tolist(),
        'model_parameters': param_count
    }
    
    output_dir = Path("heterogeneous_gnn_pyg_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(best_model_state, output_dir / "best_heterogeneous_model.pth")
    logger.info(f"💾 结果已保存到 {output_dir}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()