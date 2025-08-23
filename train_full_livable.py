#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整LIVABLE架构多分类训练脚本
使用原始LIVABLE的完整架构，不使用早停机制
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

# 添加LIVABLE代码路径
sys.path.append('LIVABLE-main/code')
from modules.model import DevignModel
from mlp_readout import MLPReadout
from data_loader.batch_graph import GGNNBatchGraph
from appnpconv import APPNPConv

try:
    import dgl
    from dgl import DGLGraph
    DGL_AVAILABLE = True
except ImportError:
    logger.error("DGL不可用，无法使用完整LIVABLE架构")
    sys.exit(1)

class FullLIVABLEDataEntry:
    """完整LIVABLE数据条目类"""
    def __init__(self, features, edges, target, sequence, edge_types):
        self.num_nodes = len(features)
        self.target = target
        self.graph = DGLGraph()
        
        # 处理特征
        self.features = torch.FloatTensor(features)
        
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        self.seq = torch.FloatTensor(sequence)
        self.edge_types = edge_types
        
        # 添加边
        sources = []
        dests = []
        etypes = []
        
        for s, edge_type, t in edges:
            if s < self.num_nodes and t < self.num_nodes:
                sources.append(s)
                dests.append(t)
                etype_number = self.get_edge_type_number(edge_type)
                etypes.append(etype_number)
        
        if sources:  # 只有当有边时才添加
            self.graph.add_edges(sources, dests, data={'etype': torch.LongTensor(etypes)})
    
    def get_edge_type_number(self, edge_type):
        """获取边类型编号"""
        if edge_type not in self.edge_types:
            self.edge_types[edge_type] = len(self.edge_types)
        return self.edge_types[edge_type]

class FullLIVABLEDataset(Dataset):
    """完整LIVABLE数据集类"""
    def __init__(self, data_path, max_seq_len=512):
        self.data = []
        self.edge_types = {}
        self.max_etype = 0
        self.max_seq_len = max_seq_len
        
        logger.info(f"📥 加载数据: {data_path}")
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        logger.info(f"📊 处理 {len(raw_data)} 个样本...")
        for item in tqdm(raw_data, desc="处理数据"):
            try:
                features = item['features']
                edges = item['structure']
                target = item['label'][0][0]
                sequence = item['sequence']
                
                # 处理序列长度
                if len(sequence) > self.max_seq_len:
                    sequence = sequence[:self.max_seq_len]
                else:
                    # 用零向量填充
                    if len(sequence) > 0:
                        feat_dim = len(sequence[0])
                        padding = [[0.0] * feat_dim] * (self.max_seq_len - len(sequence))
                        sequence = sequence + padding
                    else:
                        sequence = [[0.0] * 128] * self.max_seq_len
                
                data_entry = FullLIVABLEDataEntry(features, edges, target, sequence, self.edge_types)
                self.data.append(data_entry)
                
            except Exception as e:
                logger.warning(f"跳过无效样本: {e}")
                continue
        
        self.max_etype = len(self.edge_types)
        logger.info(f"✅ 成功加载 {len(self.data)} 个样本")
        logger.info(f"🔗 边类型数量: {self.max_etype}")
        
        # 获取序列特征维度
        if len(self.data) > 0:
            self.seq_feature_dim = len(self.data[0].seq[0])
            logger.info(f"📏 序列特征维度: {self.seq_feature_dim}")
        else:
            self.seq_feature_dim = 128
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class FullLIVABLEModel(nn.Module):
    """完整LIVABLE模型，基于原始DevignModel"""
    def __init__(self, input_dim=768, seq_input_dim=128, num_classes=14, max_edge_types=10, batch_size=16):
        super(FullLIVABLEModel, self).__init__()
        
        self.input_dim = input_dim
        self.seq_input_dim = seq_input_dim
        self.num_classes = num_classes
        self.max_edge_types = max_edge_types
        self.batch_size = batch_size
        
        # APPNP参数
        k = 16
        alpha = 0.1
        self.hidden_dim = 128
        self.hidden_dim2 = 256
        
        # APPNP层
        self.appnp = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        
        # 图分支的GRU
        self.num_layers = 1
        self.bigru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=self.num_layers, 
                           bidirectional=True, batch_first=True)
        
        # 序列分支的GRU
        self.seq_hid = 512
        self.bigru1 = nn.GRU(self.seq_input_dim, self.seq_hid, num_layers=self.num_layers,
                            bidirectional=True, batch_first=True)
        
        # 分类器
        self.MPL_layer = MLPReadout(self.hidden_dim2, num_classes)
        self.MPL_layer1 = MLPReadout(2 * self.seq_hid, num_classes)
        
        # Dropout
        self.dropout = torch.nn.Dropout(0.2)
        self.dropout1 = torch.nn.Dropout(0.2)
        
    def init_hidden(self, batch_size):
        """初始化图分支隐藏状态"""
        return torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).cuda()
    
    def init_hidden1(self, batch_size):
        """初始化序列分支隐藏状态"""
        return torch.zeros(self.num_layers * 2, batch_size, self.seq_hid).cuda()
        
    def forward(self, batch_graph, sequences):
        """前向传播"""
        batch_size = sequences.size(0)
        
        # 获取网络输入
        graph, features, edge_types = batch_graph.get_network_inputs(cuda=True)
        
        # 序列分支
        hidden1 = self.init_hidden1(batch_size)
        seq, _ = self.bigru1(sequences, hidden1)
        seq = torch.transpose(seq, 1, 2)
        seq1 = F.avg_pool1d(seq, seq.size(2)).squeeze(2)
        seq2 = F.max_pool1d(seq, seq.size(2)).squeeze(2)
        
        # 图分支
        graph = graph.to(torch.device('cuda:0'))
        
        # 解批处理
        st = batch_graph.de_batchify_graphs(features)
        
        # GRU处理
        hidden = self.init_hidden(batch_size)
        st, _ = self.bigru(st, hidden)
        
        # 重新批处理
        features = batch_graph.en_batchify_graphs(st)
        
        # 添加自环
        graph = dgl.add_self_loop(graph)
        
        # APPNP处理
        features = self.appnp(graph, features)
        
        # 再次解批处理
        st = batch_graph.de_batchify_graphs(features)
        
        # 池化
        st = torch.transpose(st, 1, 2)
        st1 = F.max_pool1d(st, st.size(2)).squeeze(2)
        st2 = F.avg_pool1d(st, st.size(2)).squeeze(2)
        
        # 分类
        graph_outputs = self.MPL_layer(self.dropout(st1 + st2))
        seq_outputs = self.MPL_layer1(self.dropout1(seq1 + seq2))
        
        # 融合输出
        final_outputs = graph_outputs + seq_outputs
        
        return final_outputs

def collate_fn(batch):
    """批处理函数"""
    batch_graph = GGNNBatchGraph()
    sequences = []
    targets = []
    
    for item in batch:
        batch_graph.add_subgraph(item.graph)
        sequences.append(item.seq)
        targets.append(item.target)
    
    # 处理序列
    sequences = torch.stack(sequences)
    targets = torch.LongTensor(targets)
    
    return batch_graph, sequences, targets

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
    output_dir = Path("full_livable_results")
    output_dir.mkdir(exist_ok=True)
    
    return device, output_dir

def load_datasets():
    """加载训练、验证和测试数据集"""
    train_dataset = FullLIVABLEDataset('livable_multiclass_data/livable_train.json')
    valid_dataset = FullLIVABLEDataset('livable_multiclass_data/livable_valid.json')
    test_dataset = FullLIVABLEDataset('livable_multiclass_data/livable_test.json')
    
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
        for batch_graph, sequences, targets in tqdm(data_loader, desc="评估中"):
            sequences = sequences.to(device)
            targets = targets.to(device)

            try:
                # 前向传播
                outputs = model(batch_graph, sequences)
                loss = criterion(outputs, targets)

                total_loss += loss.item()

                # 获取预测结果
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            except Exception as e:
                logger.warning(f"评估批次时出错: {e}")
                continue

    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )

    # 计算每个类别的准确率
    class_accuracies = {}
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    for i in range(len(conf_matrix)):
        if conf_matrix[i].sum() > 0:
            class_accuracies[i] = conf_matrix[i][i] / conf_matrix[i].sum()
        else:
            class_accuracies[i] = 0.0

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0

    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_targets, class_accuracies

def train_model(model, train_loader, valid_loader, device, num_epochs=100, learning_rate=0.001):
    """训练模型 - 不使用早停"""
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

        for batch_graph, sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            try:
                # 前向传播
                outputs = model(batch_graph, sequences)
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            except Exception as e:
                logger.warning(f"训练批次时出错: {e}")
                continue

        # 计算训练指标
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

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

    logger.info(f"🎯 训练完成！最佳验证F1: {best_valid_f1:.4f}")

    return model, training_history, best_valid_f1

def save_results(model, training_history, test_results, output_dir):
    """保存训练结果"""
    # 保存模型
    model_path = output_dir / "best_full_livable_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"💾 模型已保存: {model_path}")

    # 保存训练历史
    history_path = output_dir / "full_livable_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"📊 训练历史已保存: {history_path}")

    # 保存测试结果
    results_path = output_dir / "full_livable_final_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"📈 测试结果已保存: {results_path}")

def main():
    """主函数"""
    logger.info("🎯 开始完整LIVABLE多分类漏洞检测训练")

    # 设置环境
    device, output_dir = setup_training_environment()

    # 加载数据
    logger.info("📥 加载数据集...")
    train_dataset, valid_dataset, test_dataset = load_datasets()

    # 获取边类型数量和序列特征维度
    max_edge_types = max(train_dataset.max_etype, valid_dataset.max_etype, test_dataset.max_etype)
    seq_feature_dim = train_dataset.seq_feature_dim
    logger.info(f"🔗 最大边类型数: {max_edge_types}")
    logger.info(f"📏 序列特征维度: {seq_feature_dim}")

    # 创建数据加载器
    batch_size = 16
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=batch_size
    )

    # 创建模型
    logger.info("🏗️ 创建完整LIVABLE模型...")
    model = FullLIVABLEModel(
        input_dim=768,
        seq_input_dim=seq_feature_dim,
        num_classes=14,
        max_edge_types=max_edge_types,
        batch_size=batch_size
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 训练模型（不使用早停）
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
            'seq_input_dim': seq_feature_dim,
            'num_classes': 14,
            'max_edge_types': max_edge_types,
            'batch_size': batch_size,
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
    logger.info(f"  - 完整LIVABLE架构: ✅")

    # 等待用户输入以保持终端开启
    try:
        input("按Enter键退出...")
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except:
        logger.info("程序正常结束")

if __name__ == "__main__":
    main()
