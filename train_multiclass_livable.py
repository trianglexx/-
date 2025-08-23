#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练多分类LIVABLE模型（14个CWE类别）
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from datetime import datetime
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_training_environment():
    """设置训练环境"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建输出目录
    output_dir = Path("multiclass_training_results")
    output_dir.mkdir(exist_ok=True)
    
    return device, output_dir

def load_multiclass_data():
    """加载多分类数据"""
    
    logger.info("📥 加载多分类训练数据...")
    
    # 加载标签映射
    with open('multiclass_label_mapping.json', 'r') as f:
        mapping_info = json.load(f)
    
    logger.info(f"   类别数量: {mapping_info['num_classes']}")
    logger.info(f"   总样本数: {mapping_info['total_samples']}")
    
    # 加载数据
    data_dir = Path("livable_multiclass_data")
    datasets = {}
    
    for split in ['train', 'valid', 'test']:
        file_path = data_dir / f"livable_{split}.json"
        
        logger.info(f"   加载 {split} 数据: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        datasets[split] = data
        
        # 统计标签分布
        labels = [sample['label'][0][0] for sample in data]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"   {split}: {len(data)} 样本, 标签分布: {dict(sorted(label_counts.items()))}")
    
    return datasets, mapping_info

def create_multiclass_model(input_dim=768, hidden_dim=256, num_classes=14):
    """创建多分类模型"""
    
    class MulticlassLivableModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(MulticlassLivableModel, self).__init__()
            
            # 节点特征处理
            self.node_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # 序列特征处理
            self.sequence_encoder = nn.Sequential(
                nn.Linear(128, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            # 图级池化
            self.graph_pooling = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # 多分类器
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim // 4 + hidden_dim // 4, hidden_dim // 8),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 8, num_classes)  # 14个类别
            )
        
        def forward(self, node_features, sequence_features):
            # 处理节点特征
            node_encoded = self.node_encoder(node_features)
            
            # 图级池化（平均池化）
            graph_repr = torch.mean(node_encoded, dim=1)
            graph_repr = self.graph_pooling(graph_repr)
            
            # 处理序列特征
            seq_encoded = self.sequence_encoder(sequence_features)
            seq_repr = torch.mean(seq_encoded, dim=1)
            
            # 特征融合
            combined = torch.cat([graph_repr, seq_repr], dim=1)
            
            # 多分类
            output = self.classifier(combined)
            
            return output
    
    return MulticlassLivableModel(input_dim, hidden_dim, num_classes)

def prepare_batch_data(batch_samples, device, max_nodes=50):
    """准备批次数据"""
    
    batch_size = len(batch_samples)
    
    # 初始化张量
    node_features = torch.zeros(batch_size, max_nodes, 768, device=device)
    sequence_features = torch.zeros(batch_size, 6, 128, device=device)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i, sample in enumerate(batch_samples):
        # 节点特征
        features = sample['features']
        num_nodes = min(len(features), max_nodes)
        
        for j in range(num_nodes):
            node_features[i, j] = torch.tensor(features[j], dtype=torch.float)
        
        # 序列特征
        sequence = sample['sequence']
        seq_len = min(len(sequence), 6)
        
        for j in range(seq_len):
            sequence_features[i, j] = torch.tensor(sequence[j], dtype=torch.float)
        
        # 标签
        label = sample['label'][0][0] if sample['label'] else 0
        labels[i] = label
    
    return node_features, sequence_features, labels

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch_samples in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 准备数据
        node_features, sequence_features, labels = prepare_batch_data(batch_samples, device)
        
        # 前向传播
        outputs = model(node_features, sequence_features)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 50 == 0:
            logger.info(f"   批次 {batch_idx}/{len(dataloader)}, 损失: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

def evaluate_model(model, dataloader, criterion, device):
    """评估模型"""
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_samples in dataloader:
            # 准备数据
            node_features, sequence_features, labels = prepare_batch_data(batch_samples, device)
            
            # 前向传播
            outputs = model(node_features, sequence_features)
            loss = criterion(outputs, labels)
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

def main():
    """主训练函数"""
    
    print("🚀 开始多分类LIVABLE模型训练")
    print("14个CWE类别的漏洞类型分类")
    print("=" * 60)
    
    # 设置环境
    device, output_dir = setup_training_environment()
    
    # 加载数据
    datasets, mapping_info = load_multiclass_data()
    num_classes = mapping_info['num_classes']
    
    # 创建数据加载器
    batch_size = 16
    
    def collate_fn(batch):
        return batch
    
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"📊 数据加载器创建完成:")
    logger.info(f"   训练批次: {len(train_loader)}")
    logger.info(f"   验证批次: {len(valid_loader)}")
    logger.info(f"   测试批次: {len(test_loader)}")
    
    # 创建模型
    model = create_multiclass_model(num_classes=num_classes).to(device)
    logger.info(f"🧠 模型创建完成: {sum(p.numel() for p in model.parameters()):,} 参数")
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # 训练循环
    num_epochs = 30
    best_val_f1 = 0
    training_history = []
    
    logger.info(f"🎯 开始训练 {num_epochs} 个epoch...")
    
    for epoch in range(num_epochs):
        start_time = datetime.now()
        
        logger.info(f"\\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = evaluate_model(
            model, valid_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录历史
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_info)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_multiclass_model.pth")
            logger.info(f"   💾 保存最佳模型 (F1: {val_f1:.4f})")
        
        # 输出结果
        duration = datetime.now() - start_time
        logger.info(f"   训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        logger.info(f"   验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        logger.info(f"   验证F1: {val_f1:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"   耗时: {duration}")
        
        # 早停检查
        if epoch > 10 and val_f1 < best_val_f1 * 0.95:
            logger.info("   早停触发")
            break
    
    # 最终测试
    logger.info("\\n🎯 最终测试评估...")
    model.load_state_dict(torch.load(output_dir / "best_multiclass_model.pth"))
    test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    
    logger.info(f"📊 最终测试结果:")
    logger.info(f"   测试准确率: {test_acc:.4f}")
    logger.info(f"   测试精确率: {test_precision:.4f}")
    logger.info(f"   测试召回率: {test_recall:.4f}")
    logger.info(f"   测试F1分数: {test_f1:.4f}")
    
    # 详细分类报告
    logger.info("\\n📋 详细分类报告:")
    target_names = [f"{mapping_info['label_to_cwe'][str(i)]}" for i in range(num_classes)]
    report = classification_report(test_labels, test_preds, target_names=target_names)
    logger.info(f"\\n{report}")
    
    # 保存结果
    final_results = {
        'best_val_f1': best_val_f1,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'num_classes': num_classes,
        'total_samples': mapping_info['total_samples'],
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'classification_report': report,
        'label_mapping': mapping_info['label_to_cwe']
    }
    
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\\n" + "=" * 60)
    print("🎉 多分类训练完成!")
    print(f"📁 结果保存在: {output_dir}")
    print(f"🏆 最佳验证F1: {best_val_f1:.4f}")
    print(f"🎯 最终测试F1: {test_f1:.4f}")
    print(f"🎯 最终测试准确率: {test_acc:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
