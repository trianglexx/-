#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版步骤1: Joern CPG处理
基于VulLMGNN的成功实现，修复CPG生成和解析问题
"""

import os
import sys
import json
import argparse
import time
import subprocess
import hashlib
import re
import shutil
from typing import List, Dict, Any
from tqdm import tqdm

# 添加VulLMGNN的utils路径
sys.path.append('/root/autodl-tmp/vullmgnn/utils')
sys.path.append('/root/autodl-tmp/vullmgnn')

try:
    from utils.functions.cpg.complete_parser import parse_complete_cpg_to_nodes
    VULLMGNN_UTILS_AVAILABLE = True
    print("✅ VulLMGNN utils可用")
except ImportError:
    print("⚠️ VulLMGNN utils不可用，使用简化版本")
    VULLMGNN_UTILS_AVAILABLE = False


class JoernCPGProcessor:
    """Joern CPG处理器 - 标准接口"""

    def __init__(self, joern_path: str = "/root/autodl-tmp/vullmgnn/joern/joern-cli/"):
        """初始化Joern CPG处理器"""
        self.joern_path = joern_path
        self.fixed_processor = None

    def process_csv_to_cpg(self, csv_file: str, output_file: str, batch_size: int = 10) -> bool:
        """处理CSV文件到CPG格式"""
        try:
            # 首先将CSV转换为JSON格式
            json_file = self._convert_csv_to_json(csv_file, batch_size)

            # 确保输出目录存在
            output_dir = os.path.dirname(output_file) if output_file else "./joern_output"
            if not output_dir:
                output_dir = "./joern_output"

            # 使用内部的FixedJoernProcessor
            self.fixed_processor = FixedJoernProcessor(
                input_json=json_file,
                output_dir=output_dir,
                joern_path=self.joern_path,
                batch_size=batch_size,
                max_samples=batch_size,  # 限制样本数量用于测试
                timeout_seconds=30,
                verbose=True
            )

            # 执行处理
            self.fixed_processor.run_processing()
            return True

        except Exception as e:
            print(f"❌ Joern CPG处理失败: {e}")
            return False

    def _convert_csv_to_json(self, csv_file: str, max_samples: int) -> str:
        """将CSV文件转换为JSON格式"""
        import pandas as pd

        print(f"📊 转换CSV到JSON格式: {csv_file}")

        # 读取CSV
        df = pd.read_csv(csv_file)

        # 找到代码列
        func_col = None
        for col in ['func', 'processed_func', 'func_before', 'vul_func_with_fix']:
            if col in df.columns:
                func_col = col
                break

        if func_col is None:
            raise ValueError("CSV文件中没有找到代码列")

        # 采样数据
        if len(df) > max_samples:
            df = df.sample(max_samples)

        # 转换为JSON格式
        json_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            json_sample = {
                'id': i,
                'func': str(row[func_col]) if pd.notna(row[func_col]) else "",
                'target': int(row['target']) if 'target' in row and pd.notna(row['target']) else 0
            }
            json_data.append(json_sample)

        # 保存临时JSON文件
        json_file = "temp_joern_input.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 转换完成: {len(json_data)} 个样本")
        return json_file


class FixedJoernProcessor:
    """修复版Joern CPG处理器"""
    
    def __init__(self,
                 input_json: str,
                 output_dir: str = "./step1_output/",
                 joern_path: str = "/root/autodl-tmp/vullmgnn/joern/joern-cli/",
                 batch_size: int = 50,
                 max_samples: int = None,
                 timeout_seconds: int = 15,
                 verbose: bool = True):
        """
        初始化修复版Joern处理器
        """
        self.input_json = input_json
        self.output_dir = output_dir
        self.joern_path = joern_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建临时工作目录
        self.temp_dir = os.path.join(self.output_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 创建CPG数据输出目录
        self.cpg_data_dir = os.path.join(self.output_dir, "cpg_data")
        os.makedirs(self.cpg_data_dir, exist_ok=True)
        
        # 验证Joern工具
        self.joern_parse = os.path.join(joern_path, "joern-parse")
        self.joern_cli = os.path.join(joern_path, "joern")
        
        if not os.path.exists(self.joern_parse):
            raise FileNotFoundError(f"joern-parse未找到: {self.joern_parse}")
        if not os.path.exists(self.joern_cli):
            raise FileNotFoundError(f"joern未找到: {self.joern_cli}")
        
        # 检查Joern脚本
        self.joern_script = "/root/autodl-tmp/vullmgnn/joern/simple-cpg-extract.sc"
        if not os.path.exists(self.joern_script):
            raise FileNotFoundError(f"Joern脚本未找到: {self.joern_script}")
        
        if self.verbose:
            print("🔧 修复版Joern CPG处理器初始化")
            print(f"📂 输入文件: {self.input_json}")
            print(f"📂 输出目录: {self.output_dir}")
            print(f"📦 批次大小: {self.batch_size}")
            print(f"⏱️ 超时时间: {self.timeout_seconds}秒")
            if max_samples:
                print(f"🔢 最大样本数: {max_samples}")
    
    def log(self, message: str):
        """日志输出"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def get_code_hash(self, code: str) -> str:
        """获取代码的哈希值"""
        return hashlib.md5(code.encode('utf-8')).hexdigest()[:16]
    
    def clean_code_for_joern(self, code: str) -> str:
        """清理代码以确保Joern能正确处理 - 基于VulLMGNN的方法"""
        # 移除多余的空白字符
        code = re.sub(r'\s+', ' ', code.strip())
        
        # 确保代码不为空
        if not code or len(code.strip()) == 0:
            code = "int main() { return 0; }"
        
        # 移除特殊字符
        code = re.sub(r'[^\x00-\x7F]+', ' ', code)
        
        # 确保代码长度合理
        if len(code) > 2000:
            code = code[:2000]
        
        # 确保代码以分号或大括号结尾
        if not code.rstrip().endswith((';', '}', '{')):
            code = code.rstrip() + ';'
        
        return code
    
    def load_and_clean_data(self) -> List[Dict[str, Any]]:
        """加载并清理数据"""
        self.log("📊 加载并清理数据...")
        
        with open(self.input_json, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if self.max_samples:
            raw_data = raw_data[:self.max_samples]
        
        self.log(f"原始数据总数: {len(raw_data)}")
        
        # 清理数据
        cleaned_data = []
        
        for sample in raw_data:
            func_code = sample.get('func', '')
            target = sample.get('target', 0)
            
            # 基本过滤条件
            if (len(func_code.strip()) > 15 and 
                len(func_code) < 5000 and
                '{' in func_code and '}' in func_code and
                func_code.count('\n') < 200 and
                func_code.count(';') > 0):
                
                # 清理代码
                cleaned_code = self.clean_code_for_joern(func_code)
                
                cleaned_data.append({
                    'func': cleaned_code,
                    'original_func': func_code,
                    'target': target,
                    'metadata': sample.get('metadata', {}),
                    'index': len(cleaned_data),
                    'hash': self.get_code_hash(cleaned_code)
                })
        
        self.log(f"清洗后数据: {len(cleaned_data)}/{len(raw_data)} ({len(cleaned_data)/len(raw_data)*100:.1f}%)")
        return cleaned_data
    
    def create_batch_c_files(self, batch_data: List[Dict[str, Any]], batch_idx: int) -> tuple:
        """为批次数据创建C文件"""
        batch_dir = os.path.join(self.temp_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        
        c_files = []
        
        for i, sample in enumerate(batch_data):
            func_code = sample['func']
            file_name = f"{batch_idx}_{i}.c"
            file_path = os.path.join(batch_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(func_code)
            
            c_files.append(file_path)
        
        return batch_dir, c_files
    
    def batch_generate_cpg(self, c_files: List[str], batch_dir: str) -> List[str]:
        """批量生成CPG文件"""
        self.log(f"🔧 生成CPG文件 ({len(c_files)}个文件)...")
        
        cpg_dir = os.path.join(batch_dir, "cpg")
        os.makedirs(cpg_dir, exist_ok=True)
        
        cpg_files = []
        
        # 逐个生成CPG
        for c_file in c_files:
            file_name = os.path.basename(c_file).replace('.c', '')
            
            try:
                out_file = file_name + ".bin"
                cpg_output_path = os.path.join(cpg_dir, out_file)
                
                joern_parse_cmd = [
                    self.joern_parse,
                    c_file,
                    "--output", cpg_output_path
                ]
                
                result = subprocess.run(joern_parse_cmd, 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE, 
                                      text=True, 
                                      timeout=self.timeout_seconds,
                                      check=False)
                
                if os.path.exists(cpg_output_path) and os.path.getsize(cpg_output_path) > 100:
                    cpg_files.append(cpg_output_path)
                    
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue
        
        self.log(f"  ✅ 成功生成 {len(cpg_files)}/{len(c_files)} 个CPG文件")
        return cpg_files
    
    def extract_json_from_cpg(self, cpg_files: List[str], batch_dir: str) -> List[str]:
        """从CPG文件提取JSON数据 - 使用VulLMGNN的方法"""
        if not cpg_files:
            return []
        
        json_dir = os.path.join(batch_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        
        json_files = []
        
        # 分小批次处理
        chunk_size = 5  # 减少批次大小避免超时
        
        for i in range(0, len(cpg_files), chunk_size):
            chunk_files = cpg_files[i:i + chunk_size]
            
            # 启动Joern CLI
            joern_cmd = [self.joern_cli]
            joern_process = subprocess.Popen(joern_cmd, 
                                           stdin=subprocess.PIPE, 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE,
                                           text=True)
            
            try:
                commands = []
                chunk_json_files = []
                
                for cpg_file in chunk_files:
                    file_name = os.path.basename(cpg_file).replace('.bin', '')
                    json_file_name = f"{file_name}.json"
                    chunk_json_files.append(json_file_name)
                    
                    json_output = os.path.join(json_dir, json_file_name)
                    
                    # 使用VulLMGNN的Joern脚本
                    commands.extend([
                        f'importCpg("{os.path.abspath(cpg_file)}")',
                        f'cpg.runScript("{self.joern_script}").toString() |> "{os.path.abspath(json_output)}"',
                        'delete'
                    ])
                
                commands.append('exit')
                command_script = '\n'.join(commands)
                
                timeout = max(30, len(chunk_files) * 8)
                outs, errs = joern_process.communicate(input=command_script, timeout=timeout)
                
                # 检查生成的JSON文件
                for json_file in chunk_json_files:
                    json_path = os.path.join(json_dir, json_file)
                    if os.path.exists(json_path) and os.path.getsize(json_path) > 50:
                        json_files.append(json_path)
                
            except subprocess.TimeoutExpired:
                joern_process.kill()
                continue
            except Exception as e:
                joern_process.kill()
                self.log(f"⚠️ JSON提取失败: {e}")
                continue
        
        self.log(f"  ✅ 成功提取 {len(json_files)} 个JSON文件")
        return json_files
    
    def process_cpg_json(self, json_files: List[str], batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理CPG JSON数据"""
        cpg_results = []
        
        for json_file, sample in zip(json_files, batch_data[:len(json_files)]):
            try:
                # 读取CPG JSON数据
                with open(json_file, 'r', encoding='utf-8') as f:
                    cpg_data = json.load(f)
                
                if cpg_data and 'functions' in cpg_data and cpg_data['functions']:
                    # 使用VulLMGNN的解析器处理CPG数据
                    if VULLMGNN_UTILS_AVAILABLE:
                        try:
                            nodes = parse_complete_cpg_to_nodes(cpg_data)
                            if nodes:
                                cpg_result = {
                                    'cpg_data': cpg_data,
                                    'parsed_nodes': nodes,
                                    'target': sample['target'],
                                    'func': sample['func'],
                                    'metadata': sample['metadata'],
                                    'hash': sample['hash'],
                                    'index': sample['index'],
                                    'processing_method': 'vullmgnn_parser'
                                }
                                cpg_results.append(cpg_result)
                                continue
                        except Exception as e:
                            self.log(f"⚠️ VulLMGNN解析失败: {e}")
                    
                    # 回退到简单处理
                    cpg_result = {
                        'cpg_data': cpg_data,
                        'target': sample['target'],
                        'func': sample['func'],
                        'metadata': sample['metadata'],
                        'hash': sample['hash'],
                        'index': sample['index'],
                        'processing_method': 'simple'
                    }
                    cpg_results.append(cpg_result)
                
            except Exception as e:
                self.log(f"⚠️ JSON处理失败: {e}")
                continue
        
        return cpg_results
    
    def process_batch(self, batch_data: List[Dict[str, Any]], batch_idx: int) -> List[Dict[str, Any]]:
        """处理一个批次"""
        self.log(f"🔄 处理批次 {batch_idx + 1} ({len(batch_data)}个样本)")
        
        try:
            # 步骤1: 创建C文件
            batch_dir, c_files = self.create_batch_c_files(batch_data, batch_idx)
            
            # 步骤2: 生成CPG
            cpg_files = self.batch_generate_cpg(c_files, batch_dir)
            
            # 步骤3: 提取JSON
            json_files = self.extract_json_from_cpg(cpg_files, batch_dir)
            
            # 步骤4: 处理CPG数据
            cpg_results = self.process_cpg_json(json_files, batch_data)
            
            return cpg_results
            
        except Exception as e:
            self.log(f"  ❌ 批次处理失败: {e}")
            return []
    
    def save_cpg_data(self, cpg_results: List[Dict[str, Any]]):
        """保存CPG数据"""
        self.log("💾 保存CPG数据...")

        # 生成输出文件名
        input_basename = os.path.basename(self.input_json)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        output_filename = f"fixed_cpg_processed_{input_name_without_ext}.json"
        output_file = os.path.join(self.cpg_data_dir, output_filename)

        # 序列化CPG数据（处理CompleteNode对象）
        serializable_results = self._make_serializable(cpg_results)

        # 保存数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        self.log(f"✅ CPG数据已保存: {len(serializable_results)} 个样本 -> {output_file}")
        return output_file

    def _make_serializable(self, data):
        """将数据转换为JSON可序列化格式"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif hasattr(data, '__dict__'):
            # 处理CompleteNode等对象
            if hasattr(data, 'id') and hasattr(data, 'label'):
                # 这是一个节点对象
                properties = getattr(data, 'properties', {})
                # 递归处理properties
                serializable_properties = self._make_serializable(properties)

                return {
                    'id': str(getattr(data, 'id', '')),
                    'label': str(getattr(data, 'label', '')),
                    'code': str(getattr(data, 'code', '')),
                    'type': str(getattr(data, 'type', '')),
                    'properties': serializable_properties
                }
            else:
                # 其他对象，尝试转换为字典
                try:
                    result = {}
                    for k, v in data.__dict__.items():
                        if not k.startswith('_'):
                            try:
                                result[k] = self._make_serializable(v)
                            except:
                                result[k] = str(v)
                    return result
                except:
                    return str(data)
        elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            # 处理其他可迭代对象（如Properties）
            try:
                if hasattr(data, 'items'):
                    # 类似字典的对象
                    return {str(k): self._make_serializable(v) for k, v in data.items()}
                else:
                    # 类似列表的对象
                    return [self._make_serializable(item) for item in data]
            except:
                return str(data)
        else:
            # 基本类型或无法处理的对象
            try:
                # 尝试直接序列化
                json.dumps(data)
                return data
            except:
                return str(data)
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.log("🧹 临时文件已清理")
    
    def run_processing(self):
        """运行完整处理流程"""
        start_time = time.time()
        
        try:
            self.log("🎯 开始修复版Joern CPG处理...")
            
            # 步骤1: 加载和清理数据
            cleaned_data = self.load_and_clean_data()
            
            if not cleaned_data:
                raise Exception("清洗后没有有效数据")
            
            # 步骤2: 批量处理
            all_cpg_results = []
            total_samples = len(cleaned_data)
            
            for i in range(0, total_samples, self.batch_size):
                batch_end = min(i + self.batch_size, total_samples)
                batch_data = cleaned_data[i:batch_end]
                batch_idx = i // self.batch_size
                
                batch_start_time = time.time()
                batch_results = self.process_batch(batch_data, batch_idx)
                batch_time = time.time() - batch_start_time
                
                all_cpg_results.extend(batch_results)
                
                # 显示进度
                success_rate = len(batch_results) / len(batch_data) * 100
                speed = len(batch_results) / batch_time if batch_time > 0 else 0
                
                self.log(f"  ✅ 批次完成: {len(batch_results)}/{len(batch_data)} 成功 ({success_rate:.1f}%), 速度: {speed:.1f}样本/秒")
            
            if not all_cpg_results:
                raise Exception("没有成功处理的CPG数据")
            
            # 步骤3: 保存数据
            output_file = self.save_cpg_data(all_cpg_results)
            
            # 步骤4: 清理临时文件
            self.cleanup_temp_files()
            
            total_time = time.time() - start_time
            
            # 显示最终结果
            self.log("🎉 修复版Joern CPG处理完成！")
            self.log(f"📁 输出目录: {self.output_dir}")
            self.log(f"📄 CPG数据文件: {output_file}")
            self.log(f"⏱️ 总耗时: {total_time:.1f}秒")
            self.log(f"🚀 平均速度: {len(all_cpg_results)/total_time:.2f}样本/秒")
            
            success_rate = len(all_cpg_results) / len(cleaned_data) * 100
            self.log(f"✅ 总计处理: {len(all_cpg_results)}/{len(cleaned_data)} 样本 ({success_rate:.1f}%成功率)")
            
            return True
            
        except Exception as e:
            self.log(f"❌ 修复版Joern CPG处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="修复版步骤1: Joern CPG处理")
    
    parser.add_argument("--input", default="../output/vuln_types/all_vul_train.json", 
                       help="输入的多分类JSON文件")
    parser.add_argument("--output", default="./step1_fixed_output/", help="输出目录")
    parser.add_argument("--joern_path", default="/root/autodl-tmp/vullmgnn/joern/joern-cli/", 
                       help="Joern工具路径")
    parser.add_argument("--batch_size", type=int, default=20, help="批处理大小")
    parser.add_argument("--timeout", type=int, default=15, help="超时时间(秒)")
    parser.add_argument("--max_samples", type=int, help="最大处理样本数")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return 1
    
    # 限制样本数
    if args.max_samples:
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        limited_data = data[:args.max_samples]
        limited_file = args.input.replace('.json', f'_limited_{args.max_samples}.json')
        
        with open(limited_file, 'w') as f:
            json.dump(limited_data, f, indent=2, ensure_ascii=False)
        
        args.input = limited_file
        print(f"📊 限制样本数: {args.max_samples}")
    
    # 创建处理器并运行
    processor = FixedJoernProcessor(
        input_json=args.input,
        output_dir=args.output,
        joern_path=args.joern_path,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout,
        verbose=True
    )
    
    success = processor.run_processing()
    
    if success:
        print(f"\n🎯 修复版步骤1完成！")
        print(f"📁 输出: {args.output}")
        print(f"➡️ 下一步: 运行步骤2和步骤3")
        return 0
    else:
        print(f"\n❌ 修复版步骤1失败")
        return 1


if __name__ == "__main__":
    exit(main())
