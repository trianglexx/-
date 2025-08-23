#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版LIVABLE多分类训练脚本
使用PyTorch原生功能，避免DGL兼容性问题
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
    output_dir = Path("simple_livable_results")
    output_dir.mkdir(exist_ok=True)
    
    return device, output_dir

def load_datasets():
    """加载训练、验证和测试数据集"""
    train_dataset = SimpleLIVABLEDataset('livable_multiclass_data/livable_train.json')
    valid_dataset = SimpleLIVABLEDataset('livable_multiclass_data/livable_valid.json')
    test_dataset = SimpleLIVABLEDataset('livable_multiclass_data/livable_test.json')
    
    return train_dataset, valid_dataset, test_dataset

def create_data_loaders(train_dataset, valid_dataset, test_dataset, batch_size=16):
    """创建数据加载器"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
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
        for features, sequences, targets in tqdm(data_loader, desc="评估中"):
            features = features.to(device)
            sequences = sequences.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(features, sequences)
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

    # 计算每个类别的准确率
    class_accuracies = {}
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    for i in range(len(conf_matrix)):
        if conf_matrix[i].sum() > 0:  # 避免除零
            class_accuracies[i] = conf_matrix[i][i] / conf_matrix[i].sum()
        else:
            class_accuracies[i] = 0.0

    avg_loss = total_loss / len(data_loader)

    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_targets, class_accuracies

def train_model(model, train_loader, valid_loader, device, num_epochs=100, learning_rate=0.001):
    """训练模型 - 不使用早停，训练满100个epoch"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_valid_f1 = 0
    best_model_state = None

    training_history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_f1': []
    }

    logger.info(f"🚀 开始训练，共 {num_epochs} 个epoch（不使用早停）")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # 计算训练指标
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1, _, _, _ = evaluate_model(
            model, valid_loader, device, criterion
        )

        # 学习率调度
        scheduler.step(valid_f1)

        # 记录历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['valid_loss'].append(valid_loss)
        training_history['valid_acc'].append(valid_accuracy)
        training_history['valid_f1'].append(valid_f1)

        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}")
        logger.info(f"  验证 - Loss: {valid_loss:.4f}, Acc: {valid_accuracy:.4f}, F1: {valid_f1:.4f}")
        logger.info(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型（但不早停）
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"  ✅ 新的最佳F1分数: {best_valid_f1:.4f}")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logger.info(f"🎯 训练完成！训练了完整的 {num_epochs} 个epoch，最佳验证F1: {best_valid_f1:.4f}")

    return model, training_history, best_valid_f1

def save_results(model, training_history, test_results, output_dir):
    """保存训练结果"""
    # 保存模型
    model_path = output_dir / "best_simple_livable_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"💾 模型已保存: {model_path}")

    # 保存训练历史
    history_path = output_dir / "simple_livable_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"📊 训练历史已保存: {history_path}")

    # 保存测试结果
    results_path = output_dir / "simple_livable_final_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"📈 测试结果已保存: {results_path}")

def main():
    """主函数"""
    logger.info("🎯 开始简化版LIVABLE多分类漏洞检测训练")

    # 设置环境
    device, output_dir = setup_training_environment()

    # 加载数据
    logger.info("📥 加载数据集...")
    train_dataset, valid_dataset, test_dataset = load_datasets()

    # 创建数据加载器
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=16
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

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 训练模型（不使用早停，训练满100个epoch）
    model, training_history, best_valid_f1 = train_model(
        model, train_loader, valid_loader, device, num_epochs=100, learning_rate=0.001
    )

    # 测试模型
    logger.info("🧪 在测试集上评估模型...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_precision, test_recall, test_f1, predictions, targets, class_accuracies = evaluate_model(
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

    # 整理每个类别的准确率
    class_accuracy_dict = {}
    for i, class_name in enumerate(class_names):
        class_accuracy_dict[class_name] = class_accuracies.get(i, 0.0)

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
        'class_accuracies': class_accuracy_dict,
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_dim': 768,
            'hidden_dim': 256,
            'num_classes': 14,
            'epochs_trained': 100,
            'early_stopping': False
        }
    }

    # 打印结果
    logger.info("🎉 训练完成！")
    logger.info(f"📊 测试结果:")
    logger.info(f"  准确率: {test_accuracy:.4f}")
    logger.info(f"  精确率: {test_precision:.4f}")
    logger.info(f"  召回率: {test_recall:.4f}")
    logger.info(f"  F1分数: {test_f1:.4f}")

    # 打印每个类别的准确率
    logger.info("📈 各类别准确率:")
    for class_name, acc in class_accuracy_dict.items():
        logger.info(f"  {class_name}: {acc:.4f}")

    # 保存结果
    save_results(model, training_history, test_results, output_dir)

    logger.info("✅ 所有结果已保存完成！")

    # 保持终端开启，显示最终统计
    logger.info("🔄 训练完成，终端保持开启...")
    logger.info("📋 最终统计摘要:")
    logger.info(f"  - 训练轮数: 100 (完整训练，无早停)")
    logger.info(f"  - 最佳验证F1: {best_valid_f1:.4f}")
    logger.info(f"  - 测试F1: {test_f1:.4f}")
    logger.info(f"  - 模型参数: {total_params:,}")
    logger.info(f"  - 简化LIVABLE架构: ✅")

    # 等待用户输入以保持终端开启
    try:
        input("按Enter键退出...")
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except:
        logger.info("程序正常结束")

if __name__ == "__main__":
    main()
