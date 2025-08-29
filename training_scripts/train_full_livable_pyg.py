#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains the Full LIVABLE model architecture, rewritten using PyTorch Geometric (PyG).
Includes Early Stopping to prevent overfitting.
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

        # Graph Branch components
        self.appnp = APPNP(K=16, alpha=0.1)
        self.mlp_graph = MLPReadout(input_dim, num_classes) # Corrected: Input dim is 768

        # Sequence Branch components
        self.bigru_seq = nn.GRU(seq_input_dim, self.seq_hid, num_layers=1, bidirectional=True, batch_first=True)
        self.mlp_seq = MLPReadout(2 * self.seq_hid, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, sequence, batch = data.x, data.edge_index, data.sequence, data.batch

        # --- Sequence Branch ---
        batch_size = data.num_graphs
        # Reshape sequence from (B * L, D) to (B, L, D)
        sequence = sequence.view(batch_size, self.max_seq_len, -1)
        seq_out, _ = self.bigru_seq(sequence)
        seq_out = torch.transpose(seq_out, 1, 2)
        seq1 = F.avg_pool1d(seq_out, seq_out.size(2)).squeeze(2)
        seq2 = F.max_pool1d(seq_out, seq_out.size(2)).squeeze(2)
        seq_outputs = self.mlp_seq(self.dropout(seq1 + seq2))

        # --- Graph Branch ---
        if x.numel() > 0:
            # The GNN branch should directly process node features
            x = self.appnp(x, edge_index)
            graph_pooled = global_mean_pool(x, batch)
            graph_outputs = self.mlp_graph(self.dropout(graph_pooled))
        else:
            graph_outputs = torch.zeros_like(seq_outputs)

        return graph_outputs + seq_outputs

# --- Data and Evaluation Logic ---
def load_data_lists():
    logger.info("📥 Loading data lists...")
    with open('livable_multiclass_data/livable_train.json', 'r') as f:
        train_data = [item for item in json.load(f) if len(item['features']) > 0]
    with open('livable_multiclass_data/livable_valid.json', 'r') as f:
        valid_data = [item for item in json.load(f) if len(item['features']) > 0]
    with open('livable_multiclass_data/livable_test.json', 'r') as f:
        test_data = [item for item in json.load(f) if len(item['features']) > 0]
    logger.info(f"✅ Loaded {len(train_data)} train, {len(valid_data)} validation, {len(test_data)} test samples.")
    return train_data, valid_data, test_data

def evaluate(model, data_loader, device, criterion):
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
    _, _, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
    return avg_loss, accuracy, f1, all_predictions, all_targets

# --- Main Training Logic ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Using device: {device}")

    train_list, valid_list, test_list = load_data_lists()
    train_dataset = FullLIVABLEPygDataset(root='pyg_data/train', data_list=train_list)
    valid_dataset = FullLIVABLEPygDataset(root='pyg_data/valid', data_list=valid_list)
    test_dataset = FullLIVABLEPygDataset(root='pyg_data/test', data_list=test_list)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = FullLIVABLEPygModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs, patience, best_valid_f1, epochs_no_improve, best_model_state = 100, 10, 0, 0, None

    logger.info(f"🚀 Starting training for up to {num_epochs} epochs with early stopping (patience={patience})...")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data.y)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation check
        valid_loss, valid_acc, valid_f1, _, _ = evaluate(model, valid_loader, device, criterion)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logger.info(f"🏆 New best validation F1: {best_valid_f1:.4f}. Saving model.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"⚠️ Early stopping triggered after {epoch+1} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info("💾 Loaded best model for final evaluation.")
    else:
        logger.warning("Training did not improve. Using the last model state.")

    logger.info("📊 Evaluating best model on the test set...")
    test_loss, test_acc, test_f1, test_preds, test_targs = evaluate(model, test_loader, device, criterion)
    
    logger.info("--- FINAL MODEL PERFORMANCE ON TEST SET ---")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  F1-Score (weighted): {test_f1:.4f}")

    with open('multiclass_label_mapping.json', 'r') as f:
        label_map = json.load(f)
    class_names = [label_map['label_to_cwe'][str(i)] for i in range(14)]

    logger.info("\nClassification Report on Test Set:")
    report = classification_report(test_targs, test_preds, target_names=class_names, zero_division=0)
    print(report)

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(test_targs, test_preds)
    print(cm)

    logger.info("\nAccuracy per CWE Category (Recall):")
    # Recall is the diagonal of the normalized confusion matrix
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        logger.info(f"  {class_names[i]}: {acc:.4f}")

if __name__ == '__main__':
    main()

