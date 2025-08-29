#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异构GNN + LIVABLE融合架构
结合异构GNN的边类型建模能力与原始LIVABLE的APPNP算法
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


class APPNPLayer(nn.Module):
    """
    APPNP层的PyTorch Geometric实现
    基于原始LIVABLE的APPNP算法，但适配异构图结构
    """
    def __init__(self, k: int = 16, alpha: float = 0.1, dropout: float = 0.5):
        super(APPNPLayer, self).__init__()
        self.k = k
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: Optional[torch.Tensor] = None,
                initial_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        APPNP前向传播
        H^{l+1} = (1-α) * 消息传递 + α * H^{0}
        """
        if initial_x is None:
            initial_x = x
            
        h = x
        
        # k次APPNP迭代
        for _ in range(self.k):
            # 应用dropout到边
            if self.training:
                # 随机丢弃边
                mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.dropout.p
                masked_edge_index = edge_index[:, mask]
                if edge_type is not None:
                    masked_edge_type = edge_type[mask]
                else:
                    masked_edge_type = None
            else:
                masked_edge_index = edge_index
                masked_edge_type = edge_type
            
            # 计算度归一化的邻接矩阵消息传递
            if masked_edge_index.size(1) > 0:
                # 添加自环
                masked_edge_index, _ = add_self_loops(masked_edge_index, num_nodes=x.size(0))
                
                # 计算度
                row, col = masked_edge_index
                deg = degree(col, x.size(0), dtype=x.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                
                # 归一化消息传递
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                
                # 聚合邻居特征
                h_neighbors = torch.zeros_like(h)
                h_neighbors.index_add_(0, col, norm.view(-1, 1) * h[row])
                
                # APPNP更新：(1-α) * 邻居聚合 + α * 初始特征
                h = (1 - self.alpha) * h_neighbors + self.alpha * initial_x
            else:
                # 如果没有边，只保留初始特征
                h = initial_x
                
        return h


class HeteroAPPNPLayer(MessagePassing):
    """
    异构APPNP层：结合边类型特定的APPNP传播
    """
    def __init__(self, input_dim: int, output_dim: int, num_edge_types: int = 4,
                 k: int = 16, alpha: float = 0.1, dropout: float = 0.5):
        super(HeteroAPPNPLayer, self).__init__(aggr='add')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_edge_types = num_edge_types
        self.k = k
        self.alpha = alpha
        
        # 为每种边类型定义专用权重矩阵
        self.edge_type_weights = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(num_edge_types)
        ])
        
        # 输入投影
        if input_dim != output_dim:
            self.input_projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.input_projection = nn.Identity()
            
        # 边类型重要性权重（可学习）
        self.edge_importance = nn.Parameter(torch.ones(num_edge_types))
        
        # APPNP层
        self.appnp = APPNPLayer(k=k, alpha=alpha, dropout=dropout)
        
        # 正则化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """参数初始化"""
        gain = nn.init.calculate_gain('relu')
        for i in range(self.num_edge_types):
            nn.init.xavier_uniform_(self.edge_type_weights[i].weight, gain=gain)
            
        if hasattr(self.input_projection, 'weight'):
            nn.init.xavier_uniform_(self.input_projection.weight, gain=gain)
            
        # 初始化边类型重要性
        with torch.no_grad():
            # 基于漏洞检测先验: DFG > AST > CFG > CDG
            prior_weights = torch.tensor([1.2, 1.1, 1.4, 0.9])  # AST, CFG, DFG, CDG
            self.edge_importance.data = prior_weights
    
    def forward(self, x: torch.Tensor, edge_index_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 输入投影
        h_initial = self.input_projection(x)
        
        # 为每种边类型分别应用变换和APPNP
        edge_outputs = []
        edge_names = ['AST', 'CFG', 'DFG', 'CDG']
        edge_weights = F.softmax(self.edge_importance, dim=0)
        
        for i, edge_name in enumerate(edge_names):
            if edge_name in edge_index_dict and edge_index_dict[edge_name].size(1) > 0:
                # 应用边类型特定变换
                h_transformed = self.edge_type_weights[i](x)
                
                # APPNP传播
                h_propagated = self.appnp(h_transformed, edge_index_dict[edge_name], initial_x=h_initial)
                
                # 应用边类型权重
                h_weighted = h_propagated * edge_weights[i]
                edge_outputs.append(h_weighted)
            else:
                # 如果没有该类型的边，使用初始特征
                edge_outputs.append(h_initial * edge_weights[i])
        
        # 聚合所有边类型的输出
        if edge_outputs:
            h_final = torch.stack(edge_outputs, dim=0).sum(dim=0)
        else:
            h_final = h_initial
            
        # 正则化和激活
        h_final = self.dropout(h_final)
        h_final = self.layer_norm(h_final)
        h_final = F.gelu(h_final)
        
        return h_final


class LIVABLESequenceEncoder(nn.Module):
    """
    原始LIVABLE的序列编码器
    双向GRU + 双池化策略
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_layers: int = 2):
        super(LIVABLESequenceEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 双向GRU（仿照原始LIVABLE）
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 输出维度是双向的，所以是 hidden_dim * 2
        
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        序列编码
        Args:
            sequences: [batch_size, seq_len, input_dim]
        Returns:
            pooled_features: [batch_size, hidden_dim * 2]
        """
        # 双向GRU处理
        gru_output, _ = self.bigru(sequences)  # [batch_size, seq_len, hidden_dim * 2]
        
        # 检查维度并进行转置
        if gru_output.dim() == 3 and gru_output.size(1) > 1:
            # 转置以便池化: [batch_size, hidden_dim * 2, seq_len]
            gru_output = gru_output.transpose(1, 2)
            
            # 双池化策略（仿照原始LIVABLE）
            avg_pooled = F.avg_pool1d(gru_output, gru_output.size(2)).squeeze(2)  # [batch_size, hidden_dim * 2]
            max_pooled = F.max_pool1d(gru_output, gru_output.size(2)).squeeze(2)  # [batch_size, hidden_dim * 2]
        else:
            # 如果序列长度为1或维度不匹配，直接使用mean和max
            avg_pooled = gru_output.mean(dim=1)  # [batch_size, hidden_dim * 2]
            max_pooled = gru_output.max(dim=1)[0]  # [batch_size, hidden_dim * 2]
        
        # 相加融合（原始LIVABLE的策略）
        pooled_features = avg_pooled + max_pooled
        
        # 确保输出维度正确
        if pooled_features.dim() == 1:
            pooled_features = pooled_features.unsqueeze(0)
        
        return pooled_features


class HeteroGNNLIVABLEFusion(nn.Module):
    """
    异构GNN + LIVABLE融合模型
    结合异构图神经网络和LIVABLE的优秀算法
    """
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 num_classes: int = 14,
                 num_layers: int = 3,
                 num_edge_types: int = 4,
                 appnp_k: int = 16,
                 appnp_alpha: float = 0.1,
                 dropout: float = 0.2):
        super(HeteroGNNLIVABLEFusion, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 异构APPNP层（多层堆叠）
        self.hetero_appnp_layers = nn.ModuleList([
            HeteroAPPNPLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_edge_types=num_edge_types,
                k=appnp_k,
                alpha=appnp_alpha,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # LIVABLE序列编码器
        self.sequence_encoder = LIVABLESequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,  # 因为双向GRU输出是hidden_dim * 2
            num_layers=2
        )
        
        # 图分支MLP（仿照原始LIVABLE）
        self.graph_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 序列分支MLP（仿照原始LIVABLE）
        # 序列编码器输出维度是 hidden_dim (因为双向GRU: (hidden_dim//2) * 2 = hidden_dim)
        self.sequence_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 全局dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        """前向传播"""
        x, edge_index_dict, batch = data.x, data.edge_index_dict, data.batch
        sequence_features = data.sequence_features
        
        # === 图分支：异构APPNP处理 ===
        # 输入投影
        h = self.input_projection(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 多层异构APPNP
        for appnp_layer in self.hetero_appnp_layers:
            h_new = appnp_layer(h, edge_index_dict)
            h = h_new + h  # 残差连接
        
        # 图级别池化（双池化策略）
        graph_avg = global_mean_pool(h, batch)
        graph_max = global_max_pool(h, batch)
        graph_features = graph_avg + graph_max  # 仿照LIVABLE的相加策略
        
        # 图分支分类
        graph_outputs = self.graph_mlp(self.dropout(graph_features))
        
        # === 序列分支：LIVABLE序列编码 ===
        batch_size = int(batch.max().item() + 1)
        
        # 重新组织序列特征
        sequence_input_list = []
        for i in range(batch_size):
            mask = (batch == i)
            sample_idx = mask.nonzero(as_tuple=True)[0][0]
            seq_feat = sequence_features[sample_idx]
            sequence_input_list.append(seq_feat)
        
        # 填充到相同长度
        max_seq_len = max(seq.size(0) for seq in sequence_input_list)
        padded_sequences = []
        
        for seq in sequence_input_list:
            if seq.size(0) < max_seq_len:
                padding = torch.zeros(max_seq_len - seq.size(0), seq.size(1)).to(seq.device)
                seq = torch.cat([seq, padding], dim=0)
            padded_sequences.append(seq)
        
        sequence_input = torch.stack(padded_sequences, dim=0)
        
        # LIVABLE序列编码
        sequence_features_encoded = self.sequence_encoder(sequence_input)
        
        # 调试信息
        print(f"Debug: sequence_input shape: {sequence_input.shape}")
        print(f"Debug: sequence_features_encoded shape: {sequence_features_encoded.shape}")
        
        # 确保维度正确
        if sequence_features_encoded.dim() == 1:
            sequence_features_encoded = sequence_features_encoded.unsqueeze(0)
        elif sequence_features_encoded.size(0) == 1 and batch_size > 1:
            # 如果batch处理出现问题，重复特征以匹配batch_size
            sequence_features_encoded = sequence_features_encoded.repeat(batch_size, 1)
        
        # 序列分支分类
        sequence_outputs = self.sequence_mlp(self.dropout(sequence_features_encoded))
        
        # === 特征融合（原始LIVABLE策略）===
        # 直接相加，保持两个分支的平等贡献
        final_outputs = graph_outputs + sequence_outputs
        
        return final_outputs
    
    def get_edge_importance_weights(self):
        """获取学习到的边类型重要性权重"""
        weights_per_layer = []
        for layer in self.hetero_appnp_layers:
            layer_weights = F.softmax(layer.edge_importance, dim=0).cpu().detach().numpy()
            weights_per_layer.append(layer_weights)
        
        # 返回所有层的平均权重
        avg_weights = np.mean(weights_per_layer, axis=0)
        return avg_weights.tolist()


class HeteroLIVABLEDataset(Dataset):
    """异构LIVABLE数据集"""
    def __init__(self, data_list, max_seq_len=6):
        self.data_list = data_list
        self.max_seq_len = max_seq_len
        self.edge_type_mapping = {'AST': 0, 'CFG': 1, 'DFG': 2, 'CDG': 3}
        super(HeteroLIVABLEDataset, self).__init__()
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        sample = self.data_list[idx]
        
        # 节点特征和序列特征
        features_seq = torch.tensor(sample['features'], dtype=torch.float32)
        node_features = features_seq[-1:, :]  # 取最后一个时间步作为节点特征
        num_nodes = node_features.shape[0]
        
        # 边索引字典
        edge_index_dict = {}
        for edge_type_name in self.edge_type_mapping.keys():
            edge_index_dict[edge_type_name] = torch.tensor([], dtype=torch.long).view(2, 0)
        
        # 处理图结构
        if 'structure' in sample:
            for edge_info in sample['structure']:
                source, edge_type, target = edge_info
                if source < num_nodes and target < num_nodes and edge_type in self.edge_type_mapping:
                    if edge_index_dict[edge_type].size(1) == 0:
                        edge_index_dict[edge_type] = torch.tensor([[source], [target]], dtype=torch.long)
                    else:
                        new_edge = torch.tensor([[source], [target]], dtype=torch.long)
                        edge_index_dict[edge_type] = torch.cat([edge_index_dict[edge_type], new_edge], dim=1)
        
        # 标签
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        # 创建Data对象
        data = Data(
            x=node_features,
            edge_index_dict=edge_index_dict,
            sequence_features=features_seq,
            y=label
        )
        
        return data


def train_hetero_livable_fusion():
    """训练异构GNN + LIVABLE融合模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 使用设备: {device}")
    
    # 加载数据
    logger.info("📂 加载数据...")
    with open('livable_multiclass_data/livable_train.json', 'r') as f:
        train_data = json.load(f)
    with open('livable_multiclass_data/livable_valid.json', 'r') as f:
        valid_data = json.load(f)
    with open('livable_multiclass_data/livable_test.json', 'r') as f:
        test_data = json.load(f)
    
    # 创建数据集
    train_dataset = HeteroLIVABLEDataset(train_data, max_seq_len=6)
    valid_dataset = HeteroLIVABLEDataset(valid_data, max_seq_len=6)
    test_dataset = HeteroLIVABLEDataset(test_data, max_seq_len=6)
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = HeteroGNNLIVABLEFusion(
        input_dim=768,
        hidden_dim=256,
        num_classes=14,
        num_layers=3,
        appnp_k=16,  # 原始LIVABLE的k值
        appnp_alpha=0.1,  # 原始LIVABLE的α值
        dropout=0.2
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)
    
    # 训练参数
    num_epochs = 30
    best_valid_f1 = 0.0
    best_model = None
    patience = 10
    patience_counter = 0
    
    logger.info("🚀 开始训练异构GNN + LIVABLE融合模型")
    
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
        'model_type': 'HeteroGNN_LIVABLE_Fusion',
        'best_valid_f1': best_valid_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'edge_type_importance': edge_importance,
        'model_parameters': sum(p.numel() for p in best_model.parameters()),
        'appnp_parameters': {
            'k': 16,
            'alpha': 0.1
        },
        'classification_report': classification_report(test_labels, test_preds, 
                                                    target_names=[f'CWE-{i}' for i in [119, 20, 399, 125, 264, 200, 189, 416, 190, 362, 476, 787, 284, 254]]),
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
    }
    
    # 创建结果目录
    results_dir = Path('hetero_livable_fusion_results')
    results_dir.mkdir(exist_ok=True)
    
    # 保存结果
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(best_model.state_dict(), results_dir / 'best_fusion_model.pth')
    
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("🎉 异构GNN + LIVABLE融合模型训练完成!")
    logger.info(f"📊 测试结果:")
    logger.info(f"   - 准确率: {test_accuracy:.4f}")
    logger.info(f"   - F1分数: {test_f1:.4f}")
    logger.info(f"   - 验证最佳F1: {best_valid_f1:.4f}")
    logger.info(f"🔗 边类型重要性:")
    edge_names = ['AST', 'CFG', 'DFG', 'CDG']
    for name, weight in zip(edge_names, edge_importance):
        logger.info(f"   - {name}: {weight:.4f}")


if __name__ == "__main__":
    train_hetero_livable_fusion()