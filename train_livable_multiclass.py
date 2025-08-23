#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LIVABLE模型训练多分类漏洞检测
基于原始LIVABLE架构，适配我们的14类CWE数据
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

# 简化版本，不使用复杂的DGL功能
try:
    import dgl
    from dgl import DGLGraph
    DGL_AVAILABLE = True
except ImportError:
    logger.warning("DGL不可用，使用简化版本")
    DGL_AVAILABLE = False

class MultiClassDataEntry:
    """数据条目类，适配我们的多分类数据格式"""
    def __init__(self, features, edges, target, sequence, edge_types):
        self.num_nodes = len(features)
        self.target = target
        self.graph = DGLGraph()
        
        # 处理特征 - 从768维降到128维以适配LIVABLE
        self.features = torch.FloatTensor(features)
        features_reduced = self.features[:, :128]  # 取前128维
        
        self.graph.add_nodes(self.num_nodes, data={'features': features_reduced})
        self.seq = sequence
        self.edge_types = edge_types
        
        # 添加边
        for s, edge_type, t in edges:
            if s < self.num_nodes and t < self.num_nodes:  # 确保节点索引有效
                etype_number = self.get_edge_type_number(edge_type)
                self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})
    
    def get_edge_type_number(self, edge_type):
        """获取边类型编号"""
        if edge_type not in self.edge_types:
            self.edge_types[edge_type] = len(self.edge_types)
        return self.edge_types[edge_type]

class MultiClassDataset(Dataset):
    """多分类数据集类"""
    def __init__(self, data_path):
        self.data = []
        self.edge_types = {}
        self.max_etype = 0
        
        logger.info(f"📥 加载数据: {data_path}")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        logger.info(f"📊 处理 {len(raw_data)} 个样本...")
        for item in tqdm(raw_data, desc="处理数据"):
            try:
                features = item['features']
                edges = item['structure']  # [[src, edge_type, dst], ...]
                target = item['label'][0][0]  # 提取标签
                sequence = item['sequence']
                
                data_entry = MultiClassDataEntry(features, edges, target, sequence, self.edge_types)
                self.data.append(data_entry)
                
            except Exception as e:
                logger.warning(f"跳过无效样本: {e}")
                continue
        
        self.max_etype = len(self.edge_types)
        logger.info(f"✅ 成功加载 {len(self.data)} 个样本")
        logger.info(f"🔗 边类型数量: {self.max_etype}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class LIVABLEMultiClassModel(nn.Module):
    """LIVABLE多分类模型"""
    def __init__(self, input_dim=128, num_classes=14, max_edge_types=10):
        super(LIVABLEMultiClassModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_edge_types = max_edge_types
        
        # 基于LIVABLE的DevignModel，但修改输出类别数
        self.devign_model = DevignModel(
            input_dim=input_dim,
            output_dim=128,  # 隐藏层维度
            max_edge_types=max_edge_types,
            num_steps=8
        )
        
        # 修改最后的分类层为14类
        self.devign_model.MPL_layer = MLPReadout(256, num_classes)  # 256是hidden_dim2
        self.devign_model.MPL_layer1 = MLPReadout(1024, num_classes)  # 2 * seq_hid
        
    def forward(self, batch_data, sequences):
        """前向传播"""
        return self.devign_model(batch_data, sequences, cuda=True)

class LIVABLEBatchGraph:
    """LIVABLE兼容的批图类"""
    def __init__(self, graphs, sequences, targets):
        self.graphs = graphs
        self.sequences = sequences
        self.targets = targets
        self.batched_graph = dgl.batch(graphs)

    def get_network_inputs(self, cuda=False):
        """获取网络输入，兼容LIVABLE格式"""
        # 获取所有节点特征
        features = self.batched_graph.ndata['features']
        if cuda:
            features = features.cuda()

        return self.batched_graph, features, None

    def de_batchify_graphs(self, features):
        """将批特征分解为单个图的特征"""
        # 简化实现：直接返回特征
        return features.unsqueeze(0)  # 添加batch维度

    def en_batchify_graphs(self, features):
        """将单个图特征合并为批特征"""
        # 简化实现：移除batch维度
        return features.squeeze(0)

def collate_fn(batch):
    """批处理函数"""
    graphs = []
    sequences = []
    targets = []

    for item in batch:
        graphs.append(item.graph)
        sequences.append(item.seq)
        targets.append(item.target)

    # 处理序列 - 填充到相同长度
    max_seq_len = max(len(seq) for seq in sequences) if sequences else 1
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_seq_len:
            seq = seq + [0] * (max_seq_len - len(seq))  # 用0填充
        elif len(seq) > max_seq_len:
            seq = seq[:max_seq_len]  # 截断
        padded_sequences.append(seq)

    # 创建LIVABLE兼容的批图
    batch_graph = LIVABLEBatchGraph(graphs, padded_sequences, targets)

    return batch_graph, torch.FloatTensor(padded_sequences), torch.LongTensor(targets)

def setup_training_environment():
    """设置训练环境"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建输出目录
    output_dir = Path("livable_training_results")
    output_dir.mkdir(exist_ok=True)
    
    return device, output_dir

def load_datasets():
    """加载训练、验证和测试数据集"""
    train_dataset = MultiClassDataset('livable_multiclass_data/livable_train.json')
    valid_dataset = MultiClassDataset('livable_multiclass_data/livable_valid.json')
    test_dataset = MultiClassDataset('livable_multiclass_data/livable_test.json')
    
    return train_dataset, valid_dataset, test_dataset

def create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=16):
    """创建数据加载器"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # 避免多进程问题
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, valid_loader, test_loader

def evaluate_model(model, data_loader, device, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_graph, sequences, targets in tqdm(data_loader, desc="评估中"):
            sequences = sequences.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(batch_graph, sequences)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_targets

def train_model(model, train_loader, valid_loader, device, num_epochs=50, learning_rate=0.0001):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    best_valid_f1 = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    training_history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_f1': []
    }

    logger.info(f"🚀 开始训练，共 {num_epochs} 个epoch")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_graph, sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_graph, sequences)
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # 计算训练指标
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1, _, _ = evaluate_model(
            model, valid_loader, device, criterion
        )

        # 记录历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['valid_loss'].append(valid_loss)
        training_history['valid_acc'].append(valid_accuracy)
        training_history['valid_f1'].append(valid_f1)

        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}")
        logger.info(f"  验证 - Loss: {valid_loss:.4f}, Acc: {valid_accuracy:.4f}, F1: {valid_f1:.4f}")

        # 早停检查
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  ✅ 新的最佳F1分数: {best_valid_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  ⏹️ 早停触发，最佳F1: {best_valid_f1:.4f}")
                break

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, training_history, best_valid_f1

def save_results(model, training_history, test_results, output_dir):
    """保存训练结果"""
    # 保存模型
    model_path = output_dir / "best_livable_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"💾 模型已保存: {model_path}")

    # 保存训练历史
    history_path = output_dir / "livable_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"📊 训练历史已保存: {history_path}")

    # 保存测试结果
    results_path = output_dir / "livable_final_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"📈 测试结果已保存: {results_path}")

def main():
    """主函数"""
    logger.info("🎯 开始LIVABLE多分类漏洞检测训练")

    # 设置环境
    device, output_dir = setup_training_environment()

    # 加载数据
    logger.info("📥 加载数据集...")
    train_dataset, valid_dataset, test_dataset = load_datasets()

    # 获取边类型数量
    max_edge_types = max(train_dataset.max_etype, valid_dataset.max_etype, test_dataset.max_etype)
    logger.info(f"🔗 最大边类型数: {max_edge_types}")

    # 创建数据加载器
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=16
    )

    # 创建模型
    logger.info("🏗️ 创建LIVABLE模型...")
    model = LIVABLEMultiClassModel(
        input_dim=128,
        num_classes=14,
        max_edge_types=max_edge_types
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 训练模型
    model, training_history, best_valid_f1 = train_model(
        model, train_loader, valid_loader, device, num_epochs=50, learning_rate=0.0001
    )

    # 测试模型
    logger.info("🧪 在测试集上评估模型...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1, predictions, targets = evaluate_model(
        model, test_loader, device, criterion
    )

    # 生成分类报告
    with open('multiclass_label_mapping.json', 'r') as f:
        label_mapping = json.load(f)

    class_names = [label_mapping['label_to_cwe'][str(i)] for i in range(14)]
    classification_rep = classification_report(
        targets, predictions, target_names=class_names, output_dict=True, zero_division=0
    )

    # 混淆矩阵
    conf_matrix = confusion_matrix(targets, predictions)

    # 整理测试结果
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'best_valid_f1': best_valid_f1,
        'classification_report': classification_rep,
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': class_names,
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_dim': 128,
            'num_classes': 14,
            'max_edge_types': max_edge_types
        }
    }

    # 打印结果
    logger.info("🎉 训练完成！")
    logger.info(f"📊 测试结果:")
    logger.info(f"  准确率: {test_accuracy:.4f}")
    logger.info(f"  精确率: {test_precision:.4f}")
    logger.info(f"  召回率: {test_recall:.4f}")
    logger.info(f"  F1分数: {test_f1:.4f}")

    # 保存结果
    save_results(model, training_history, test_results, output_dir)

    logger.info("✅ 所有结果已保存完成！")

if __name__ == "__main__":
    main()
