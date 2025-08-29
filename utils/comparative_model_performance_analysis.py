#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能对比分析脚本
比较不同架构模型在多分类漏洞检测任务上的性能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

def collect_model_results():
    """收集所有模型的结果"""
    results = {}
    
    # 1. 异构GNN (PyG版本)
    try:
        with open('heterogeneous_gnn_pyg_results/results.json', 'r') as f:
            hetero_results = json.load(f)
        results['Heterogeneous GNN'] = {
            'test_accuracy': hetero_results['test_accuracy'],
            'test_f1': hetero_results['test_f1'],
            'best_valid_f1': hetero_results['best_valid_f1'],
            'model_parameters': hetero_results['model_parameters'],
            'edge_importance': hetero_results['edge_type_importance'],
            'architecture': 'Heterogeneous Graph Neural Network',
            'key_features': ['Dynamic edge weights', 'Multi-edge types', 'Attention mechanism']
        }
    except FileNotFoundError:
        print("⚠️ 异构GNN结果文件未找到，将使用当前训练结果")
        
    # 2. Simple LIVABLE
    try:
        with open('simple_livable_results/simple_livable_final_results.json', 'r') as f:
            simple_results = json.load(f)
        results['Simple LIVABLE'] = {
            'test_accuracy': simple_results['test_accuracy'],
            'test_f1': simple_results['test_f1'],
            'best_valid_f1': simple_results['best_valid_f1'],
            'model_parameters': simple_results['model_info']['total_params'],
            'architecture': 'Simplified LIVABLE (Graph + Sequence)',
            'key_features': ['Graph encoding', 'Sequence modeling', 'Feature fusion']
        }
    except FileNotFoundError:
        print("⚠️ Simple LIVABLE结果文件未找到")
        
    # 3. SADE Loss模型
    try:
        with open('sade_training_results/sade_final_results.json', 'r') as f:
            sade_results = json.load(f)
        results['SADE Loss'] = {
            'test_accuracy': sade_results['test_accuracy'],
            'test_f1': sade_results['test_f1'],
            'best_valid_f1': sade_results['best_val_f1'],
            'model_parameters': sade_results['model_parameters'],
            'architecture': 'LIVABLE with Self-Adaptive Loss',
            'key_features': ['Adaptive loss weighting', 'Class imbalance handling', 'Dynamic adjustment']
        }
    except FileNotFoundError:
        print("⚠️ SADE结果文件未找到")
        
    # 4. 标准多分类模型
    try:
        with open('multiclass_training_results/final_results.json', 'r') as f:
            multi_results = json.load(f)
        results['Standard Multiclass'] = {
            'test_accuracy': multi_results['test_accuracy'],
            'test_f1': multi_results['test_f1'],
            'best_valid_f1': multi_results['best_val_f1'],
            'model_parameters': multi_results['model_parameters'],
            'architecture': 'Standard LIVABLE Multiclass',
            'key_features': ['Basic graph features', 'Standard cross-entropy loss']
        }
    except FileNotFoundError:
        print("⚠️ 标准多分类结果文件未找到")
    
    return results

def analyze_performance(results):
    """分析模型性能"""
    print("🔍 模型性能对比分析")
    print("=" * 80)
    
    # 创建性能对比DataFrame
    performance_data = []
    for model_name, metrics in results.items():
        performance_data.append({
            'Model': model_name,
            'Test Accuracy': metrics['test_accuracy'],
            'Test F1': metrics['test_f1'],
            'Best Valid F1': metrics['best_valid_f1'],
            'Parameters (M)': metrics['model_parameters'] / 1_000_000,
            'Architecture': metrics['architecture']
        })
    
    df = pd.DataFrame(performance_data)
    
    # 排序：按测试F1分数排序
    df = df.sort_values('Test F1', ascending=False)
    
    print("\n📊 模型性能排名 (按测试F1分数排序):")
    print("-" * 80)
    for idx, row in df.iterrows():
        print(f"{row.name + 1:2d}. {row['Model']:<20} | "
              f"Acc: {row['Test Accuracy']:.4f} | "
              f"F1: {row['Test F1']:.4f} | "
              f"Params: {row['Parameters (M)']:.2f}M")
    
    # 性能分析
    print(f"\n🏆 最佳模型分析:")
    best_model = df.iloc[0]
    print(f"   模型: {best_model['Model']}")
    print(f"   架构: {best_model['Architecture']}")
    print(f"   测试准确率: {best_model['Test Accuracy']:.4f}")
    print(f"   测试F1分数: {best_model['Test F1']:.4f}")
    print(f"   参数量: {best_model['Parameters (M)']:.2f}M")
    
    # 效率分析 (F1/参数比)
    df['Efficiency'] = df['Test F1'] / df['Parameters (M)']
    most_efficient = df.loc[df['Efficiency'].idxmax()]
    print(f"\n⚡ 最高效模型:")
    print(f"   模型: {most_efficient['Model']}")
    print(f"   效率指标: {most_efficient['Efficiency']:.6f} (F1/M参数)")
    
    return df, results

def create_comparison_plots(df, results):
    """创建性能对比图表"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 准确率vs F1分数散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['Test Accuracy'], df['Test F1'], 
                         s=df['Parameters (M)'] * 50, 
                         alpha=0.7, c=range(len(df)), cmap='viridis')
    
    for i, row in df.iterrows():
        ax1.annotate(row['Model'], 
                    (row['Test Accuracy'], row['Test F1']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    ax1.set_xlabel('Test Accuracy')
    ax1.set_ylabel('Test F1 Score')
    ax1.set_title('Performance Comparison\n(Bubble size = Parameters)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 参数量对比条形图
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(df)), df['Parameters (M)'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(df))))
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Parameters (Millions)')
    ax2.set_title('Model Complexity Comparison')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                        for name in df['Model']], rotation=45)
    
    # 在条形上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}M', ha='center', va='bottom')
    
    # 3. F1分数对比
    ax3 = axes[1, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax3.bar(df['Model'], df['Test F1'], 
                   color=colors[:len(df)])
    ax3.set_ylabel('Test F1 Score')
    ax3.set_title('F1 Score Comparison')
    ax3.set_xticklabels(df['Model'], rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # 4. 效率分析 (F1/参数比)
    ax4 = axes[1, 1]
    df_sorted_eff = df.sort_values('Efficiency', ascending=True)
    bars = ax4.barh(range(len(df_sorted_eff)), df_sorted_eff['Efficiency'],
                    color=plt.cm.plasma(np.linspace(0, 1, len(df_sorted_eff))))
    ax4.set_xlabel('Efficiency (F1 Score / Million Parameters)')
    ax4.set_title('Model Efficiency Comparison')
    ax4.set_yticks(range(len(df_sorted_eff)))
    ax4.set_yticklabels(df_sorted_eff['Model'])
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.6f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_edge_importance(results):
    """分析异构GNN的边类型重要性"""
    if 'Heterogeneous GNN' in results and 'edge_importance' in results['Heterogeneous GNN']:
        print(f"\n🔗 异构GNN边类型重要性分析:")
        print("-" * 50)
        
        edge_names = ['AST', 'CFG', 'DFG', 'CDG']
        edge_weights = results['Heterogeneous GNN']['edge_importance']
        
        # 创建边类型重要性分析
        for name, weight in zip(edge_names, edge_weights):
            bar = "█" * int(weight * 50)  # 可视化权重
            print(f"   {name:<4}: {weight:.4f} |{bar}")
        
        # 找出最重要的边类型
        most_important_idx = np.argmax(edge_weights)
        most_important_edge = edge_names[most_important_idx]
        print(f"\n   🎯 最重要的边类型: {most_important_edge} (权重: {edge_weights[most_important_idx]:.4f})")
        
        # 边类型重要性的理论分析
        print(f"\n   💡 理论分析:")
        edge_analysis = {
            'AST': '语法结构树 - 代码的语法关系',
            'CFG': '控制流图 - 程序执行路径',
            'DFG': '数据流图 - 变量依赖关系',
            'CDG': '控制依赖图 - 控制语句依赖'
        }
        
        for name, weight in zip(edge_names, edge_weights):
            print(f"     {name}: {edge_analysis[name]} (学习权重: {weight:.4f})")

def generate_detailed_report(df, results):
    """生成详细的对比报告"""
    report = []
    report.append("# 🎯 多分类漏洞检测模型性能对比报告")
    report.append("=" * 60)
    report.append("")
    
    # 概述
    report.append("## 📊 实验概述")
    report.append("")
    report.append("本报告对比了四种不同架构的深度学习模型在14类CWE漏洞检测任务上的性能表现:")
    report.append("")
    
    for i, (model_name, model_data) in enumerate(results.items(), 1):
        report.append(f"{i}. **{model_name}**: {model_data['architecture']}")
        key_features = ', '.join(model_data['key_features'])
        report.append(f"   - 核心特性: {key_features}")
        report.append("")
    
    # 性能排名
    report.append("## 🏆 性能排名")
    report.append("")
    report.append("| 排名 | 模型名称 | 测试准确率 | 测试F1 | 验证F1 | 参数量(M) |")
    report.append("|------|----------|-----------|--------|--------|----------|")
    
    for idx, row in df.iterrows():
        report.append(f"| {idx+1} | {row['Model']} | {row['Test Accuracy']:.4f} | "
                     f"{row['Test F1']:.4f} | {row['Best Valid F1']:.4f} | {row['Parameters (M)']:.2f} |")
    
    report.append("")
    
    # 详细分析
    report.append("## 🔍 详细性能分析")
    report.append("")
    
    best_model = df.iloc[0]
    worst_model = df.iloc[-1]
    
    report.append(f"### 🥇 最佳性能模型: {best_model['Model']}")
    report.append("")
    report.append(f"- **测试准确率**: {best_model['Test Accuracy']:.4f}")
    report.append(f"- **测试F1分数**: {best_model['Test F1']:.4f}")
    report.append(f"- **参数量**: {best_model['Parameters (M)']:.2f}M")
    report.append("")
    
    if 'Heterogeneous GNN' in results:
        hetero_data = results['Heterogeneous GNN']
        if 'edge_importance' in hetero_data:
            report.append("#### 🔗 边类型重要性分析 (仅异构GNN)")
            edge_names = ['AST', 'CFG', 'DFG', 'CDG']
            edge_weights = hetero_data['edge_importance']
            
            for name, weight in zip(edge_names, edge_weights):
                report.append(f"- **{name}**: {weight:.4f}")
            report.append("")
    
    # 效率分析
    df_eff = df.copy()
    df_eff['Efficiency'] = df_eff['Test F1'] / df_eff['Parameters (M)']
    most_efficient = df_eff.loc[df_eff['Efficiency'].idxmax()]
    
    report.append(f"### ⚡ 最高效模型: {most_efficient['Model']}")
    report.append("")
    report.append(f"- **效率指标**: {most_efficient['Efficiency']:.6f} (F1分数/百万参数)")
    report.append(f"- **测试F1**: {most_efficient['Test F1']:.4f}")
    report.append(f"- **参数量**: {most_efficient['Parameters (M)']:.2f}M")
    report.append("")
    
    # 关键发现
    report.append("## 🎯 关键发现")
    report.append("")
    
    performance_gap = best_model['Test F1'] - worst_model['Test F1']
    report.append(f"1. **性能差距**: 最佳与最差模型F1分数差距为 {performance_gap:.4f}")
    
    avg_accuracy = df['Test Accuracy'].mean()
    avg_f1 = df['Test F1'].mean()
    report.append(f"2. **平均性能**: 平均准确率 {avg_accuracy:.4f}, 平均F1分数 {avg_f1:.4f}")
    
    max_params = df['Parameters (M)'].max()
    min_params = df['Parameters (M)'].min()
    report.append(f"3. **模型复杂度**: 参数量范围从 {min_params:.2f}M 到 {max_params:.2f}M")
    
    report.append("")
    report.append("## 💡 结论与建议")
    report.append("")
    
    if best_model['Model'] == 'Heterogeneous GNN':
        report.append("异构图神经网络在多分类漏洞检测任务上表现最佳，其动态边权重调整机制")
        report.append("和多类型边建模能力为其带来了显著的性能优势。")
    else:
        report.append(f"{best_model['Model']} 在此任务上表现最佳，展现了其架构设计的有效性。")
    
    report.append("")
    report.append("**建议**:")
    report.append("- 对于追求最高性能的场景，推荐使用性能最佳的模型")
    report.append("- 对于资源受限的环境，推荐使用效率最高的模型")
    report.append("- 可以考虑模型集成来进一步提升性能")
    
    return "\n".join(report)

def main():
    """主函数"""
    print("🚀 开始模型性能对比分析...")
    
    # 收集结果
    results = collect_model_results()
    
    if not results:
        print("❌ 没有找到模型结果文件")
        return
    
    print(f"✅ 成功加载 {len(results)} 个模型的结果")
    
    # 性能分析
    df, results = analyze_performance(results)
    
    # 边类型重要性分析
    analyze_edge_importance(results)
    
    # 创建对比图表
    print(f"\n📊 生成性能对比图表...")
    create_comparison_plots(df, results)
    
    # 生成详细报告
    print(f"\n📝 生成详细对比报告...")
    detailed_report = generate_detailed_report(df, results)
    
    # 保存报告
    with open('comprehensive_model_comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    
    # 保存数据
    df.to_csv('model_performance_comparison.csv', index=False)
    
    print(f"\n🎉 分析完成!")
    print(f"📁 文件已保存:")
    print(f"   - comprehensive_model_comparison_report.md (详细报告)")
    print(f"   - model_performance_comparison.csv (性能数据)")
    print(f"   - model_performance_comparison.png (对比图表)")

if __name__ == "__main__":
    main()