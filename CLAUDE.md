# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a vulnerability detection multiclass classification system based on Joern CPG + GraphCodeBERT + LIVABLE architecture, supporting 14 CWE type classifications. It combines code property graph analysis with deep learning for software vulnerability detection.

## Key Commands

### Data Generation
```bash
# Create multiclass dataset from scratch
python create_multiclass_from_scratch.py
```

### Model Training
```bash
# Standard multiclass training
python train_multiclass_livable.py

# Training with SADE adaptive loss function
python train_with_sade_loss.py

# Simple LIVABLE model training
python train_simple_livable.py

# Training with early stopping
python train_with_early_stopping.py

# Training on full LIVABLE architecture
python train_full_livable_pyg.py
```

### Model Evaluation
```bash
# Evaluate final trained model
python evaluate_final_model.py

# Standard training and evaluation pipeline
python train_eval_standard.py
```

### Original LIVABLE Framework
```bash
# Run original LIVABLE model (in LIVABLE-main/code/)
cd LIVABLE-main/code
python main_sta.py --dataset multi --input_dir ../our_word2vec_multi
```

## Architecture Overview

### Core Components

1. **Data Processing Pipeline**
   - `tools/joern_cpg_processor.py`: Joern CPG (Code Property Graph) processing
   - `tools/graphcodebert_processor.py`: GraphCodeBERT embedding generation
   - Raw data → Joern CPG → GraphCodeBERT embeddings → LIVABLE format

2. **Model Architectures**
   - **LIVABLE (original)**: Full graph neural network with differentiated propagation
   - **Simplified LIVABLE**: Streamlined version for multiclass classification
   - **SADE Loss**: Self-adaptive loss function for handling class imbalance

3. **Training Components**
   - Node feature processing (768-dim GraphCodeBERT embeddings)
   - Graph structure encoding (AST + CFG + DFG + CDG)
   - Sequence feature processing for temporal patterns
   - Multi-head attention mechanisms

### Data Structure

- **Input Features**: 768-dimensional GraphCodeBERT embeddings per node
- **Graph Structure**: Multi-edge types (AST, CFG, DFG, CDG)
- **Labels**: 14 CWE vulnerability types (0-13)
- **Dataset Split**: 70% train / 15% validation / 15% test

### Key Directories

- `livable_multiclass_data/`: Training/validation/test splits in LIVABLE format
- `all_vul_full_processed/`: Complete processed data with Joern and GraphCodeBERT outputs
- `multiclass_training_results/`: Trained model checkpoints and results
- `LIVABLE-main/`: Original LIVABLE implementation with configurations
- `tools/`: Data processing utilities

## Dependencies

Core requirements (install via pip):
```bash
pip install torch transformers scikit-learn numpy pandas
```

For original LIVABLE framework:
- torch (==1.9.0)
- dgl (==0.7.2) 
- numpy (==1.22.3)
- pandas (==1.4.1)

## Training Configuration

### Standard Parameters
- Input dimension: 768 (GraphCodeBERT)
- Hidden dimension: 256
- Batch size: 16
- Learning rate: 0.001
- Max nodes per graph: 50
- Sequence length: 6 timesteps

### SADE Loss Parameters
- alpha: 1.0 (base weight)
- beta: 2.0 (class balance weight)  
- gamma: 0.5 (adaptive weight)

## CWE Classification Mapping

The system classifies 14 major CWE vulnerability types:
- CWE-119: Buffer Overflow (most frequent)
- CWE-20: Input Validation 
- CWE-399: Resource Management
- CWE-125: Out-of-bounds Read
- And 10 other CWE types

Label mappings are stored in `multiclass_label_mapping.json`.

## Performance Notes

- Dataset exhibits severe class imbalance (CWE-119 dominates)
- Typical accuracy: ~27-28% (realistic for imbalanced multiclass)
- SADE loss provides marginal improvements for tail classes
- Focus on F1-score rather than raw accuracy for evaluation

## Model Files

Trained models are saved in respective result directories:
- `best_multiclass_model.pth`: Best performing multiclass model
- `final_simple_model.pth`: Simple LIVABLE variant
- Training history and detailed results in JSON format