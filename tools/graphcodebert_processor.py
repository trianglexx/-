#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphCodeBERT处理器
将CodeBERT替换为GraphCodeBERT，提供更好的代码理解能力
"""

# Hugging Face镜像配置
import os
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/root/.cache/huggingface'

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
from transformers import (
    RobertaTokenizer, 
    RobertaModel,
    AutoTokenizer,
    AutoModel
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphCodeBERTProcessor:
    """GraphCodeBERT处理器"""
    
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        """
        初始化GraphCodeBERT处理器
        
        Args:
            model_name: GraphCodeBERT模型名称
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🤖 初始化GraphCodeBERT处理器...")
        logger.info(f"   模型: {model_name}")
        logger.info(f"   设备: {self.device}")
        
        # 加载tokenizer和model
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载GraphCodeBERT模型"""
        try:
            logger.info("📥 加载GraphCodeBERT tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            logger.info("📥 加载GraphCodeBERT model...")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_safetensors=True  # 使用safetensors格式避免torch.load问题
            ).to(self.device)
            
            self.model.eval()
            
            logger.info("✅ GraphCodeBERT模型加载成功")
            logger.info(f"   词汇表大小: {self.tokenizer.vocab_size}")
            logger.info(f"   隐藏层维度: {self.model.config.hidden_size}")
            
        except Exception as e:
            logger.error(f"❌ GraphCodeBERT模型加载失败: {e}")
            raise
    
    def encode_code(self, code: str, max_length: int = 512) -> np.ndarray:
        """
        使用GraphCodeBERT编码代码
        
        Args:
            code: 源代码字符串
            max_length: 最大序列长度
            
        Returns:
            代码嵌入向量 (768维)
        """
        try:
            # 预处理代码
            code = self._preprocess_code(code)
            
            # Tokenize
            inputs = self.tokenizer(
                code,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS] token的嵌入作为代码表示
                code_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return code_embedding.flatten()
            
        except Exception as e:
            logger.warning(f"⚠️ 代码编码失败: {e}")
            # 返回零向量作为fallback
            return np.zeros(768, dtype=np.float32)
    
    def encode_code_with_dfg(self, code: str, dfg_edges: List[Tuple] = None, 
                            max_length: int = 512) -> np.ndarray:
        """
        使用GraphCodeBERT编码代码和数据流图
        
        Args:
            code: 源代码字符串
            dfg_edges: 数据流图边列表 [(source, target, relation)]
            max_length: 最大序列长度
            
        Returns:
            增强的代码嵌入向量 (768维)
        """
        try:
            # 预处理代码
            code = self._preprocess_code(code)
            
            # 如果有DFG信息，构建增强输入
            if dfg_edges:
                # 构建DFG增强的输入
                enhanced_input = self._build_dfg_enhanced_input(code, dfg_edges)
            else:
                enhanced_input = code
            
            # Tokenize
            inputs = self.tokenizer(
                enhanced_input,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS] token的嵌入作为代码表示
                code_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return code_embedding.flatten()
            
        except Exception as e:
            logger.warning(f"⚠️ DFG增强编码失败: {e}")
            # 回退到普通编码
            return self.encode_code(code, max_length)
    
    def _preprocess_code(self, code: str) -> str:
        """预处理代码"""
        if not code or not isinstance(code, str):
            return ""
        
        # 移除过多的空白字符
        code = ' '.join(code.split())
        
        # 限制长度
        if len(code) > 2000:
            code = code[:2000]
        
        return code
    
    def _build_dfg_enhanced_input(self, code: str, dfg_edges: List[Tuple]) -> str:
        """构建DFG增强的输入"""
        # 简化版本：将DFG信息作为注释添加到代码中
        dfg_info = []
        for edge in dfg_edges[:10]:  # 限制DFG边数量
            if len(edge) >= 3:
                source, target, relation = edge[0], edge[1], edge[2]
                dfg_info.append(f"{source}->{target}({relation})")
        
        if dfg_info:
            dfg_comment = "// DFG: " + ", ".join(dfg_info)
            enhanced_input = f"{dfg_comment}\n{code}"
        else:
            enhanced_input = code
        
        return enhanced_input
    
    def batch_encode_codes(self, codes: List[str], max_length: int = 512) -> np.ndarray:
        """
        批量编码代码
        
        Args:
            codes: 代码列表
            max_length: 最大序列长度
            
        Returns:
            代码嵌入矩阵 [batch_size, 768]
        """
        embeddings = []
        
        logger.info(f"🔄 批量编码 {len(codes)} 个代码样本...")
        
        for i, code in enumerate(codes):
            if i % 100 == 0:
                logger.info(f"   进度: {i}/{len(codes)}")
            
            embedding = self.encode_code(code, max_length)
            embeddings.append(embedding)
        
        logger.info("✅ 批量编码完成")
        return np.array(embeddings)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
            "hidden_size": self.model.config.hidden_size if self.model else 0,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 512),
            "device": str(self.device)
        }

def migrate_codebert_to_graphcodebert(input_file: str, output_file: str):
    """
    将CodeBERT嵌入迁移到GraphCodeBERT
    
    Args:
        input_file: 包含CodeBERT嵌入的输入文件
        output_file: GraphCodeBERT嵌入的输出文件
    """
    logger.info("🔄 开始CodeBERT到GraphCodeBERT的迁移...")
    
    # 初始化GraphCodeBERT处理器
    processor = GraphCodeBERTProcessor()
    
    # 加载数据
    logger.info(f"📥 加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"   样本数量: {len(data)}")
    
    # 处理每个样本
    updated_data = []
    for i, sample in enumerate(data):
        if i % 50 == 0:
            logger.info(f"   处理进度: {i}/{len(data)}")
        
        # 获取原始代码
        code = sample.get('func', '') or sample.get('original_code', '')
        
        if code:
            # 使用GraphCodeBERT重新编码
            new_embedding = processor.encode_code(code)
            
            # 更新嵌入
            sample['code_embedding'] = new_embedding.tolist()
            sample['embedding_model'] = 'graphcodebert-base'
            sample['embedding_dim'] = len(new_embedding)
        else:
            logger.warning(f"   样本 {i} 没有代码内容，跳过")
        
        updated_data.append(sample)
    
    # 保存更新后的数据
    logger.info(f"💾 保存更新数据: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ 迁移完成!")
    
    # 输出统计信息
    model_info = processor.get_model_info()
    logger.info("📊 模型信息:")
    for key, value in model_info.items():
        logger.info(f"   {key}: {value}")

def test_graphcodebert():
    """测试GraphCodeBERT处理器"""
    logger.info("🧪 测试GraphCodeBERT处理器...")
    
    # 初始化处理器
    processor = GraphCodeBERTProcessor()
    
    # 测试代码
    test_code = """
    int vulnerable_function(char* input) {
        char buffer[100];
        strcpy(buffer, input);  // Buffer overflow vulnerability
        return strlen(buffer);
    }
    """
    
    # 编码测试
    embedding = processor.encode_code(test_code)
    
    logger.info(f"✅ 测试成功!")
    logger.info(f"   嵌入维度: {embedding.shape}")
    logger.info(f"   嵌入范围: [{embedding.min():.4f}, {embedding.max():.4f}]")
    logger.info(f"   嵌入均值: {embedding.mean():.4f}")
    
    return True

if __name__ == "__main__":
    # 测试GraphCodeBERT处理器
    test_graphcodebert()
