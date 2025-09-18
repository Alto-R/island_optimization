#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专注分析可再生能源、储能和LNG成本的绝对值和占比这六个变量与总成本下降的关系
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置图形参数 - Nature期刊风格
plt.rcParams.update({
    'font.family': 'Arial',  # Nature期刊偏好的字体
    'font.size': 14,  # 较大的字体用于清晰显示
    'axes.linewidth': 1.2,  # 坐标轴线宽
    'xtick.major.width': 1.2,  # X轴刻度线宽
    'ytick.major.width': 1.2,  # Y轴刻度线宽
    'figure.dpi': 300,  # 图像分辨率
    'savefig.dpi': 300,  # 保存图像分辨率
    'savefig.bbox': 'tight'  # 紧凑布局保存
})

print("开始六变量与总成本下降关系分析...")

# 1. 数据加载和预处理 - 专注六个核心变量
def load_and_prepare_data():
    """加载并预处理数据，计算六个核心变量和总成本下降率"""
    # 加载成本基础数据（技术进步前）
    cost_base_file = Path("island_cost_summary_2050.csv")
    cost_base = pd.read_csv(cost_base_file)
    print(f"基础情景数据形状: {cost_base.shape}")

    # 加载技术进步情景数据（技术进步后）
    cost_future_file = Path("island_cost_summary_future_2050.csv")
    cost_future = pd.read_csv(cost_future_file)
    print(f"技术进步情景数据形状: {cost_future.shape}")

    # 确保两个数据集有相同的岛屿（按经纬度匹配）
    cost_analysis = cost_base.copy()

    # 三大成本类型
    cost_categories = ['renewable_cost', 'storage_cost', 'lng_cost']

    # 计算总成本（基础情景和技术进步情景）
    total_cost_base = cost_base[cost_categories].sum(axis=1)  # 技术进步前总成本
    total_cost_future = cost_future[cost_categories].sum(axis=1)  # 技术进步后总成本

    # 计算总成本下降率（正值表示成本下降）
    valid_total_mask = (total_cost_base > 1e-6) & (total_cost_base.notna())  # 避免除零
    cost_analysis['total_cost_reduction_rate'] = 0.0  # 初始化
    cost_analysis.loc[valid_total_mask, 'total_cost_reduction_rate'] = (
        (total_cost_base.loc[valid_total_mask] - total_cost_future.loc[valid_total_mask]) /
        total_cost_base.loc[valid_total_mask] * 100  # 转换为百分比
    )

    # 计算六个核心变量
    # 1-3: 三大成本绝对值（基础情景）
    for category in cost_categories:
        cost_analysis[f'{category}_absolute'] = cost_base[category]  # 绝对成本值

    # 4-6: 三大成本占比（基础情景）
    for category in cost_categories:
        cost_analysis[f'{category}_proportion'] = 0.0  # 初始化
        cost_analysis.loc[valid_total_mask, f'{category}_proportion'] = (
            cost_base.loc[valid_total_mask, category] / total_cost_base.loc[valid_total_mask]  # 成本占比
        )

    # 过滤有意义的数据点：总成本下降超过1%
    significant_reduction = cost_analysis['total_cost_reduction_rate'] > 0.1
    data_filtered = cost_analysis[significant_reduction].copy()
    
    print(f"\n数据处理结果:")
    print(f"总岛屿数: {len(cost_analysis)}")
    print(f"有显著成本下降的岛屿数: {len(data_filtered)}")
    print(f"平均总成本下降率: {data_filtered['total_cost_reduction_rate'].mean():.2f}%")

    # 六个核心变量的统计信息
    print(f"\n六个核心变量统计:")
    for category in cost_categories:
        abs_var = f'{category}_absolute'
        prop_var = f'{category}_proportion'
        print(f"{category}:")
        print(f"  绝对值均值: {data_filtered[abs_var].mean():.2f}")
        print(f"  占比均值: {data_filtered[prop_var].mean():.3f}")

    return data_filtered

# 2. 六变量相关性分析
def analyze_six_variable_correlations(data):
    """分析六个核心变量与总成本下降率的相关性"""
    print("\n=== 六变量相关性分析 ===")

    # 定义六个核心变量
    six_variables = [
        'renewable_cost_absolute',    # 可再生能源成本绝对值
        'storage_cost_absolute',      # 储能成本绝对值
        'lng_cost_absolute',          # LNG成本绝对值
        'renewable_cost_proportion',  # 可再生能源成本占比
        'storage_cost_proportion',    # 储能成本占比
        'lng_cost_proportion'         # LNG成本占比
    ]

    target_variable = 'total_cost_reduction_rate'  # 总成本下降率

    # 计算每个变量与总成本下降率的Pearson相关系数
    correlations = {}
    p_values = {}

    print("各变量与总成本下降率的相关性:")
    print("-" * 60)

    for var in six_variables:
        if var in data.columns:
            # 计算Pearson相关系数和p值
            corr_coef, p_val = pearsonr(data[var], data[target_variable])
            correlations[var] = corr_coef
            p_values[var] = p_val

            # 相关性强度判断
            if abs(corr_coef) >= 0.7:
                strength = "强"
            elif abs(corr_coef) >= 0.4:
                strength = "中等"
            elif abs(corr_coef) >= 0.2:
                strength = "弱"
            else:
                strength = "极弱"

            # 显著性判断
            significance = "显著" if p_val < 0.05 else "不显著"

            print(f"{var:<25}: r={corr_coef:>6.3f}, p={p_val:>7.4f} ({strength}, {significance})")

    # 创建相关性结果DataFrame
    correlation_results = pd.DataFrame({
        'Variable': six_variables,
        'Correlation': [correlations.get(var, 0) for var in six_variables],
        'P_Value': [p_values.get(var, 1) for var in six_variables],
        'Significant': [p_values.get(var, 1) < 0.05 for var in six_variables]
    })

    # 按相关系数绝对值排序
    correlation_results['Abs_Correlation'] = correlation_results['Correlation'].abs()
    correlation_results = correlation_results.sort_values('Abs_Correlation', ascending=False)

    print(f"\n按重要性排序的变量:")
    print("-" * 60)
    for i, row in correlation_results.iterrows():
        print(f"{i+1}. {row['Variable']:<25}: |r|={row['Abs_Correlation']:.3f}")

    return correlations, correlation_results

# 3. 多元线性回归分析
def perform_multiple_regression(data):
    """使用六个变量进行多元线性回归分析"""
    print("\n=== 多元线性回归分析 ===")

    # 定义自变量（六个核心变量）
    feature_variables = [
        'renewable_cost_absolute',    # 可再生能源成本绝对值
        'storage_cost_absolute',      # 储能成本绝对值
        'lng_cost_absolute',          # LNG成本绝对值
        'renewable_cost_proportion',  # 可再生能源成本占比
        'storage_cost_proportion',    # 储能成本占比
        'lng_cost_proportion'         # LNG成本占比
    ]

    target_variable = 'total_cost_reduction_rate'  # 因变量：总成本下降率

    # 准备数据
    X = data[feature_variables].copy()  # 自变量矩阵
    y = data[target_variable].copy()    # 因变量向量

    # 检查数据完整性
    print(f"回归分析数据形状: X={X.shape}, y={y.shape}")
    print(f"缺失值检查: X缺失值={X.isnull().sum().sum()}, y缺失值={y.isnull().sum()}")

    # 标准化特征（用于系数比较）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 标准化后的特征矩阵

    # 多元线性回归
    model = LinearRegression()
    model.fit(X_scaled, y)  # 训练模型

    # 预测和模型评估
    y_pred = model.predict(X_scaled)  # 预测值
    r2 = r2_score(y, y_pred)          # 决定系数R²
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # 均方根误差

    print(f"\n模型性能评估:")
    print(f"决定系数 R2 = {r2:.4f}")  # R2越接近1模型越好
    print(f"均方根误差 RMSE = {rmse:.4f}")

    # 回归系数分析
    coefficients_df = pd.DataFrame({
        'Variable': feature_variables,           # 变量名
        'Coefficient': model.coef_,             # 标准化回归系数
        'Abs_Coefficient': np.abs(model.coef_)  # 系数绝对值（用于重要性排序）
    })

    # 按重要性排序（绝对值大的更重要）
    coefficients_df = coefficients_df.sort_values('Abs_Coefficient', ascending=False)

    print(f"\n回归系数分析 (按重要性排序):")
    print("-" * 70)
    print(f"{'变量':<25} {'系数':<10} {'绝对值':<10} {'影响方向':<10}")
    print("-" * 70)

    for _, row in coefficients_df.iterrows():
        direction = "正向" if row['Coefficient'] > 0 else "负向"
        print(f"{row['Variable']:<25} {row['Coefficient']:>9.4f} {row['Abs_Coefficient']:>9.4f} {direction:>9}")

    print(f"\n截距项: {model.intercept_:.4f}")

    # 特征重要性解释
    print(f"\n回归结果解释:")
    most_important = coefficients_df.iloc[0]
    print(f"最重要变量: {most_important['Variable']}")
    print(f"该变量每增加1个标准差，总成本下降率变化: {most_important['Coefficient']:.4f}%")

    return model, coefficients_df, scaler, r2, rmse

# 4. 专门的六变量可视化函数
def create_six_variable_visualizations(data, correlations, coefficients_df):
    """创建专门针对六个核心变量的可视化图表"""
    print("\n=== 创建六变量可视化图表 ===")

    # 定义六个核心变量
    six_variables = [
        'renewable_cost_absolute',    # 可再生能源成本绝对值
        'storage_cost_absolute',      # 储能成本绝对值
        'lng_cost_absolute',          # LNG成本绝对值
        'renewable_cost_proportion',  # 可再生能源成本占比
        'storage_cost_proportion',    # 储能成本占比
        'lng_cost_proportion'         # LNG成本占比
    ]

    target_variable = 'total_cost_reduction_rate'  # 总成本下降率

    # === 图1: 散点图矩阵 - 展示六变量与总成本下降的关系 ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2行3列布局
    axes = axes.flatten()  # 展平为一维数组

    # 定义颜色
    colors = ['forestgreen', 'orange', 'steelblue', 'red', 'purple', 'brown']

    for i, (var, color) in enumerate(zip(six_variables, colors)):
        ax = axes[i]

        # 创建散点图
        scatter = ax.scatter(data[var], data[target_variable],
                           alpha=0.6,           # 透明度
                           c=color,            # 颜色
                           s=30,               # 点大小
                           edgecolors='black', # 边框颜色
                           linewidth=0.5)      # 边框宽度

        # 添加拟合线
        z = np.polyfit(data[var], data[target_variable], 1)  # 一次拟合
        p = np.poly1d(z)
        ax.plot(data[var], p(data[var]),
               color='red',         # 拟合线颜色
               linestyle='--',      # 虚线
               alpha=0.8,          # 透明度
               linewidth=2)        # 线宽

        # 设置坐标轴标签
        ax.set_xlabel(var.replace('_', ' ').title(), fontsize=12)  # X轴标签
        ax.set_ylabel('Cost Reduction Rate (%)', fontsize=12)     # Y轴标签

        # 添加相关系数文本
        corr = correlations.get(var, 0)
        ax.text(0.05, 0.95, f'r = {corr:.3f}',              # 相关系数文本
               transform=ax.transAxes,      # 使用轴坐标
               fontsize=11,                # 字体大小
               fontweight='bold',          # 加粗
               bbox=dict(boxstyle='round,pad=0.3',  # 文本框样式
                        facecolor='white',         # 背景色
                        alpha=0.8))               # 透明度

        # 设置网格
        ax.grid(True, alpha=0.3)  # 网格透明度

        # 设置刻度
        ax.tick_params(axis='both', which='major', labelsize=10)  # 刻度字体大小

    plt.tight_layout(pad=2.0)  # 调整子图间距
    plt.savefig('six_variable_scatter_matrix.png',
               dpi=300,              # 分辨率
               bbox_inches='tight')  # 紧凑保存
    plt.show()

    # === 图2: 相关性热图 ===
    fig, ax = plt.subplots(figsize=(10, 8))

    # 准备相关性矩阵数据
    correlation_matrix_data = []
    variables_for_heatmap = six_variables + [target_variable]

    # 计算所有变量间的相关性
    corr_matrix = data[variables_for_heatmap].corr()  # 计算相关性矩阵

    # 创建热图
    im = ax.imshow(corr_matrix.values,           # 相关性矩阵值
                   cmap='RdBu_r',               # 颜色映射（红蓝反转）
                   aspect='auto',               # 自动调整长宽比
                   vmin=-1, vmax=1)            # 颜色范围[-1,1]

    # 设置刻度标签
    ax.set_xticks(range(len(variables_for_heatmap)))  # X轴刻度位置
    ax.set_yticks(range(len(variables_for_heatmap)))  # Y轴刻度位置
    ax.set_xticklabels([var.replace('_', '\n') for var in variables_for_heatmap],
                      rotation=45, ha='right', fontsize=10)  # X轴标签
    ax.set_yticklabels([var.replace('_', '\n') for var in variables_for_heatmap],
                      fontsize=10)  # Y轴标签

    # 添加数值标注
    for i in range(len(variables_for_heatmap)):
        for j in range(len(variables_for_heatmap)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',  # 相关系数值
                          ha="center", va="center",                # 居中对齐
                          color="black", fontsize=9,              # 字体颜色和大小
                          fontweight='bold')                      # 加粗

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)  # 颜色条
    cbar.set_label('Correlation Coefficient', fontsize=12)  # 颜色条标签

    plt.tight_layout()
    plt.savefig('six_variable_correlation_heatmap.png',
               dpi=300, bbox_inches='tight')
    plt.show()

    # === 图3: 回归系数条形图 ===
    fig, ax = plt.subplots(figsize=(12, 8))

    # 准备数据
    variables = coefficients_df['Variable'].tolist()
    coefficients = coefficients_df['Coefficient'].tolist()

    # 创建条形图
    bars = ax.bar(range(len(variables)),          # X轴位置
                  coefficients,                   # 条形高度
                  color=['forestgreen' if c > 0 else 'red' for c in coefficients],  # 正负不同颜色
                  alpha=0.8,                     # 透明度
                  edgecolor='black',             # 边框颜色
                  linewidth=1)                   # 边框宽度

    # 设置坐标轴
    ax.set_xticks(range(len(variables)))         # X轴刻度位置
    ax.set_xticklabels([var.replace('_', '\n') for var in variables],
                      rotation=45, ha='right', fontsize=11)  # X轴标签
    ax.set_ylabel('Regression Coefficient', fontsize=12)     # Y轴标签

    # 添加数值标签
    for i, (bar, coef) in enumerate(zip(bars, coefficients)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,    # X位置
               height + 0.01 if height >= 0 else height - 0.03,  # Y位置
               f'{coef:.3f}',                       # 数值文本
               ha='center', va='bottom' if height >= 0 else 'top',  # 对齐方式
               fontsize=10, fontweight='bold')      # 字体设置

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # 零基准线

    # 设置网格
    ax.grid(True, alpha=0.3, axis='y')  # 仅Y轴网格

    plt.tight_layout()
    plt.savefig('six_variable_regression_coefficients.png',
               dpi=300, bbox_inches='tight')
    plt.show()

    print("六变量可视化图表已生成:")
    print("1. six_variable_scatter_matrix.png - 散点图矩阵")
    print("2. six_variable_correlation_heatmap.png - 相关性热图")
    print("3. six_variable_regression_coefficients.png - 回归系数图")

# 5. 简化的主分析函数
def main_six_variable_analysis():
    """专注于六变量分析的主流程"""
    print("="*60)
    print("六变量与总成本下降关系分析")
    print("="*60)

    # 1. 加载和预处理数据
    data = load_and_prepare_data()

    # 2. 六变量相关性分析
    correlations, correlation_results = analyze_six_variable_correlations(data)

    # 3. 多元线性回归分析
    model, coefficients_df, scaler, r2, rmse = perform_multiple_regression(data)

    # 4. 创建可视化图表
    create_six_variable_visualizations(data, correlations, coefficients_df)

    # 5. 分类分析：寻找成本下降最多的情况
    category_results = analyze_cost_reduction_by_categories(data)

    # 5. 生成简化分析报告
    generate_six_variable_report(data, correlation_results, coefficients_df, r2, rmse)

    return data, correlations, model, coefficients_df

# 6. 简化的分析报告生成
def generate_six_variable_report(data, correlation_results, coefficients_df, r2, rmse):
    """生成专门针对六变量分析的报告"""
    print("\n" + "="*70)
    print("六变量与总成本下降关系分析报告")
    print("="*70)

    # 基本统计信息
    system_performance = data.groupby('system_name').agg({
        'total_cost_change_rate': ['mean', 'std', 'count'],
        'renewable_cost_change_rate': 'mean',
        'storage_cost_change_rate': 'mean',
        'lng_cost_change_rate': 'mean',
        'renewable_cost_ratio': 'mean',
        'storage_cost_ratio': 'mean',
        'lng_cost_ratio': 'mean'
    }).round(3)

    print("各能源系统类型性能:")
    print(system_performance)

    # 统计检验 - 不同系统类型间的显著性差异
    system_groups = [group['total_cost_change_rate'].values for name, group in data.groupby('system_name')]
    if len(system_groups) > 1:
        f_stat, p_value = stats.f_oneway(*system_groups)
        print(f"\nANOVA检验 - F统计量: {f_stat:.3f}, p值: {p_value:.6f}")

    return system_performance

# 7. 可视化函数
def create_comprehensive_visualizations(data, feature_importance, category_importance, system_performance):
    """创建全面的可视化图表"""
    print("\n=== 创建综合可视化图表 ===")

    # 图1: 三大成本类型综合分析 (2x2 subplot)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 各成本类型占比分布
    ax1 = axes[0, 0]
    cost_ratios = ['renewable_cost_ratio', 'storage_cost_ratio', 'lng_cost_ratio']
    colors = ['forestgreen', 'orange', 'steelblue']

    for i, (ratio, color) in enumerate(zip(cost_ratios, colors)):
        ax1.hist(data[ratio], bins=30, alpha=0.7, color=color, label=ratio.replace('_cost_ratio', '').title())

    ax1.set_xlabel('Cost Ratio', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Distribution of Cost Ratios', fontsize=13, fontweight='bold')

    # 子图2: 能源系统类型与成本下降
    ax2 = axes[0, 1]
    system_names = data['system_name'].unique()
    system_data = [data[data['system_name'] == name]['total_cost_change_rate'] for name in system_names]

    bp2 = ax2.boxplot(system_data, labels=[name.replace(' ', '\n') for name in system_names], patch_artist=True)
    colors_box = plt.cm.Set3(np.linspace(0, 1, len(system_names)))
    for patch, color in zip(bp2['boxes'], colors_box):
        patch.set_facecolor(color)

    ax2.set_ylabel('Total Cost Reduction Rate (%)', fontsize=12)
    ax2.set_xlabel('Energy System Type', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_title('Cost Reduction by System Type', fontsize=13, fontweight='bold')

    # 子图3: 特征重要性（按类型分组）
    ax3 = axes[1, 0]
    category_colors = {'renewable': 'forestgreen', 'storage': 'orange', 'lng': 'steelblue'}

    bars = ax3.bar(range(len(category_importance)), category_importance.values,
                   color=[category_colors[cat] for cat in category_importance.index], alpha=0.8)

    ax3.set_xticks(range(len(category_importance)))
    ax3.set_xticklabels([cat.title() for cat in category_importance.index], fontsize=11)
    ax3.set_ylabel('Total Feature Importance', fontsize=12)
    ax3.set_title('Feature Importance by Cost Category', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, value in zip(bars, category_importance.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 子图4: 三维散点图（前三个最重要特征）
    ax4 = axes[1, 1]
    top_features = feature_importance.head(3)['feature'].tolist()

    # 使用前两个最重要特征的2D散点图
    x_col, y_col = top_features[0], top_features[1]
    scatter = ax4.scatter(data[x_col], data[y_col], c=data['total_cost_change_rate'],
                         cmap='viridis', alpha=0.6, s=30)

    ax4.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax4.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax4.set_title('Top Features vs Cost Reduction', fontsize=13, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Total Cost Reduction Rate (%)', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_cost_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图2: 地理分布与成本结构
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 地理区域分析
    regions = data['region'].unique()
    region_colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))

    # 子图1: 各区域储能成本分布
    ax1 = axes[0, 0]
    for i, region in enumerate(regions):
        region_data = data[data['region'] == region]
        ax1.scatter(region_data['storage_cost'], region_data['total_cost_change_rate'],
                   alpha=0.6, label=region, c=[region_colors[i]], s=40)

    ax1.set_xlabel('Storage Cost (absolute)', fontsize=12)
    ax1.set_ylabel('Total Cost Reduction Rate (%)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Storage Cost vs Reduction by Region', fontsize=13, fontweight='bold')

    # 子图2: 各区域可再生能源成本分布
    ax2 = axes[0, 1]
    for i, region in enumerate(regions):
        region_data = data[data['region'] == region]
        ax2.scatter(region_data['renewable_cost'], region_data['total_cost_change_rate'],
                   alpha=0.6, label=region, c=[region_colors[i]], s=40)

    ax2.set_xlabel('Renewable Cost (absolute)', fontsize=12)
    ax2.set_ylabel('Total Cost Reduction Rate (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Renewable Cost vs Reduction by Region', fontsize=13, fontweight='bold')

    # 子图3: 各区域LNG成本分布
    ax3 = axes[1, 0]
    for i, region in enumerate(regions):
        region_data = data[data['region'] == region]
        ax3.scatter(region_data['lng_cost'], region_data['total_cost_change_rate'],
                   alpha=0.6, label=region, c=[region_colors[i]], s=40)

    ax3.set_xlabel('LNG Cost (absolute)', fontsize=12)
    ax3.set_ylabel('Total Cost Reduction Rate (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('LNG Cost vs Reduction by Region', fontsize=13, fontweight='bold')

    # 子图4: 三大成本比例三角图（简化版）
    ax4 = axes[1, 1]
    # 使用散点图显示三项成本占比的关系
    scatter = ax4.scatter(data['renewable_cost_ratio'], data['storage_cost_ratio'],
                         c=data['lng_cost_ratio'], cmap='plasma', alpha=0.7, s=50)

    ax4.set_xlabel('Renewable Cost Ratio', fontsize=12)
    ax4.set_ylabel('Storage Cost Ratio', fontsize=12)
    ax4.set_title('Cost Structure Triangle', fontsize=13, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('LNG Cost Ratio', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('geographic_cost_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 图3: 相关性热图
    fig, ax = plt.subplots(figsize=(12, 10))

    corr_cols = [
        'renewable_cost', 'storage_cost', 'lng_cost',
        'renewable_cost_ratio', 'storage_cost_ratio', 'lng_cost_ratio',
        'total_cost_change_rate', 'renewable_cost_change_rate',
        'storage_cost_change_rate', 'lng_cost_change_rate'
    ]

    correlation_matrix = data[corr_cols].corr()

    # 创建热图
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)

    ax.set_title('Comprehensive Cost Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('comprehensive_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. 简化的主分析函数
def main_six_variable_analysis():
    """专注于六变量分析的主流程"""
    print("="*60)
    print("六变量与总成本下降关系分析")
    print("="*60)

    # 1. 加载和预处理数据
    data = load_and_prepare_data()

    # 2. 六变量相关性分析
    correlations, correlation_results = analyze_six_variable_correlations(data)

    # 3. 多元线性回归分析
    model, coefficients_df, scaler, r2, rmse = perform_multiple_regression(data)

    # 4. 创建可视化图表
    create_six_variable_visualizations(data, correlations, coefficients_df)

    # 5. 分类分析：寻找成本下降最多的情况
    category_results = analyze_cost_reduction_by_categories(data)

    # 5. 生成简化分析报告
    generate_six_variable_report(data, correlation_results, coefficients_df, r2, rmse)

    return data, correlations, model, coefficients_df

# 6. 简化的分析报告生成
def generate_six_variable_report(data, correlation_results, coefficients_df, r2, rmse):
    """生成专门针对六变量分析的报告"""
    print("\n" + "="*70)
    print("六变量与总成本下降关系分析报告")
    print("="*70)

    # 基本统计信息
    print(f"\n1. 数据概览:")
    print(f"   分析样本数: {len(data)}")
    print(f"   平均总成本下降率: {data['total_cost_reduction_rate'].mean():.2f}%")
    print(f"   总成本下降率标准差: {data['total_cost_reduction_rate'].std():.2f}%")

    # 六个变量的基本统计
    print(f"\n2. 六个核心变量统计:")
    six_variables = [
        ('renewable_cost_absolute', '可再生能源成本绝对值'),
        ('storage_cost_absolute', '储能成本绝对值'),
        ('lng_cost_absolute', 'LNG成本绝对值'),
        ('renewable_cost_proportion', '可再生能源成本占比'),
        ('storage_cost_proportion', '储能成本占比'),
        ('lng_cost_proportion', 'LNG成本占比')
    ]

    for var, desc in six_variables:
        mean_val = data[var].mean()
        std_val = data[var].std()
        print(f"   {desc}: 均值={mean_val:.3f}, 标准差={std_val:.3f}")

    # 相关性分析结果
    print(f"\n3. 相关性分析结果 (按重要性排序):")
    for i, row in correlation_results.iterrows():
        significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
        print(f"   {i+1}. {row['Variable']:<25}: r={row['Correlation']:>7.3f}, p={row['P_Value']:>7.4f} {significance}")

    # 回归分析结果
    print(f"\n4. 多元回归分析结果:")
    print(f"   模型解释度 R2 = {r2:.4f}")
    print(f"   均方根误差 RMSE = {rmse:.4f}")
    print(f"   回归系数 (按重要性排序):")

    for i, row in coefficients_df.iterrows():
        direction = "促进下降" if row['Coefficient'] > 0 else "抑制下降"
        print(f"   {i+1}. {row['Variable']:<25}: {row['Coefficient']:>7.4f} ({direction})")

    # 关键发现
    print(f"\n5. 关键发现:")

    # 最重要的相关因子
    top_corr = correlation_results.iloc[0]
    print(f"   最强相关变量: {top_corr['Variable']} (r={top_corr['Correlation']:.3f})")

    # 最重要的回归因子
    top_regr = coefficients_df.iloc[0]
    print(f"   最重要回归变量: {top_regr['Variable']} (系数={top_regr['Coefficient']:.3f})")

    # 绝对值vs占比的重要性比较
    abs_vars = correlation_results[correlation_results['Variable'].str.contains('absolute')]
    prop_vars = correlation_results[correlation_results['Variable'].str.contains('proportion')]

    abs_mean_corr = abs_vars['Abs_Correlation'].mean()
    prop_mean_corr = prop_vars['Abs_Correlation'].mean()

    print(f"   绝对值变量平均相关性: {abs_mean_corr:.3f}")
    print(f"   占比变量平均相关性: {prop_mean_corr:.3f}")

    if abs_mean_corr > prop_mean_corr:
        print(f"   结论: 绝对成本值对总成本下降的影响更重要")
    else:
        print(f"   结论: 成本占比对总成本下降的影响更重要")

    # 各成本类型的重要性
    print(f"\n6. 各成本类型重要性:")
    for category in ['renewable', 'storage', 'lng']:
        abs_var = f'{category}_cost_absolute'
        prop_var = f'{category}_cost_proportion'

        abs_corr = correlation_results[correlation_results['Variable'] == abs_var]['Abs_Correlation'].iloc[0] if len(correlation_results[correlation_results['Variable'] == abs_var]) > 0 else 0
        prop_corr = correlation_results[correlation_results['Variable'] == prop_var]['Abs_Correlation'].iloc[0] if len(correlation_results[correlation_results['Variable'] == prop_var]) > 0 else 0

        total_importance = abs_corr + prop_corr
        print(f"   {category.title()}: 总重要性={total_importance:.3f} (绝对值={abs_corr:.3f}, 占比={prop_corr:.3f})")

    print(f"\n分析完成！")
    print(f"详细可视化图表已保存为PNG文件。")

# 专门分析有LNG成本的岛屿
def analyze_lng_islands_separately(data, var_name, var_desc, target_variable):
    """专门分析有LNG成本的岛屿，提供更精细的分析"""
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')

    # 筛选有LNG成本的岛屿
    lng_islands = data[data[var_name] > 0].copy()
    lng_count = len(lng_islands)

    print(f"有LNG成本的岛屿数: {lng_count}")
    print(f"占总样本比例: {lng_count/len(data)*100:.1f}%")

    if lng_count < 10:
        print("警告: 有LNG成本的岛屿数量过少（<10），无法进行可靠的统计分析")
        return

    # 计算LNG岛屿的基本统计
    print(f"\n有LNG成本岛屿的{var_desc}统计:")
    print(f"范围: {lng_islands[var_name].min():.3f} - {lng_islands[var_name].max():.3f}")
    print(f"均值: {lng_islands[var_name].mean():.3f}")
    print(f"标准差: {lng_islands[var_name].std():.3f}")
    print(f"中位数: {lng_islands[var_name].median():.3f}")

    # 相关性分析：仅针对有LNG成本的岛屿
    from scipy.stats import pearsonr
    corr_coef, p_val = pearsonr(lng_islands[var_name], lng_islands[target_variable])

    print(f"\n相关性分析（仅有LNG成本的岛屿）:")
    print(f"Pearson相关系数: r = {corr_coef:.4f}")
    print(f"P值: p = {p_val:.4f}")

    if p_val < 0.05:
        print(f"结论: 相关性显著 ({'正相关' if corr_coef > 0 else '负相关'})")
    else:
        print(f"结论: 相关性不显著")

    # 分析LNG成本分布特征
    print(f"\nLNG成本分布特征分析:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("百分位数分布:")
    for p in percentiles:
        val = lng_islands[var_name].quantile(p/100)
        print(f"  {p}%分位数: {val:.1f}")

    # 重新定义高成本组的划分方法，基于更明显的分界点
    # 方法1：基于90%分位数作为高成本阈值
    q90_threshold = lng_islands[var_name].quantile(0.90)
    high_cost_count_q90 = len(lng_islands[lng_islands[var_name] >= q90_threshold])

    # 方法2：基于传统IQR方法
    q75 = lng_islands[var_name].quantile(0.75)
    q25 = lng_islands[var_name].quantile(0.25)
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    high_cost_count_iqr = len(lng_islands[lng_islands[var_name] > outlier_threshold])

    # 方法3：基于对数分布，找到明显的跳跃点
    non_zero_values = lng_islands[lng_islands[var_name] > 0][var_name].sort_values()
    if len(non_zero_values) > 10:
        # 计算相邻值的比率，找到最大跳跃
        ratios = non_zero_values.iloc[1:].values / non_zero_values.iloc[:-1].values
        max_jump_idx = ratios.argmax()
        jump_threshold = non_zero_values.iloc[max_jump_idx]
        high_cost_count_jump = len(lng_islands[lng_islands[var_name] >= jump_threshold])
    else:
        jump_threshold = q90_threshold
        high_cost_count_jump = high_cost_count_q90

    print(f"\n极端值分析 - 三种划分方法比较:")
    print(f"方法1 - 90%分位数阈值: {q90_threshold:.1f}")
    print(f"  高成本岛屿数量: {high_cost_count_q90} ({high_cost_count_q90/lng_count*100:.1f}%)")

    print(f"方法2 - IQR异常值阈值: {outlier_threshold:.1f}")
    print(f"  高成本岛屿数量: {high_cost_count_iqr} ({high_cost_count_iqr/lng_count*100:.1f}%)")

    print(f"方法3 - 最大跳跃点阈值: {jump_threshold:.1f}")
    print(f"  高成本岛屿数量: {high_cost_count_jump} ({high_cost_count_jump/lng_count*100:.1f}%)")

    # 方法4：基于实际业务意义的固定阈值（100）
    fixed_threshold = 100
    high_cost_count_fixed = len(lng_islands[lng_islands[var_name] >= fixed_threshold])

    print(f"方法4 - 固定阈值(100): {fixed_threshold:.1f}")
    print(f"  高成本岛屿数量: {high_cost_count_fixed} ({high_cost_count_fixed/lng_count*100:.1f}%)")

    # 选择最优的划分方法：选择固定阈值100，更有实际意义
    selected_threshold = fixed_threshold
    high_cost_count = high_cost_count_fixed
    selected_method = "固定阈值法(100)"

    print(f"\n选用方法: {selected_method}")
    print(f"最终高成本阈值: {selected_threshold:.1f}")
    print(f"高成本岛屿数量: {high_cost_count} ({high_cost_count/lng_count*100:.1f}%)")

    # 三分位数分析（仅针对有LNG成本的岛屿）
    if lng_count >= 15:  # 确保每组至少有约5个样本
        q33_lng = lng_islands[var_name].quantile(0.33)
        q67_lng = lng_islands[var_name].quantile(0.67)

        print(f"\n三分位数分类（仅有LNG成本的岛屿）:")
        print(f"Low: ≤ {q33_lng:.3f}")
        print(f"Medium: {q33_lng:.3f} < value < {q67_lng:.3f}")
        print(f"High: ≥ {q67_lng:.3f}")

        # 额外进行极端值分组分析
        print(f"\n基于{selected_method}的分组分析:")
        print(f"常规成本组 (<{selected_threshold:.1f}): {lng_count - high_cost_count}个岛屿")
        print(f"高成本组 (≥{selected_threshold:.1f}): {high_cost_count}个岛屿")

        # 创建分类
        lng_islands['lng_category'] = 'Medium'
        lng_islands.loc[lng_islands[var_name] <= q33_lng, 'lng_category'] = 'Low'
        lng_islands.loc[lng_islands[var_name] >= q67_lng, 'lng_category'] = 'High'

        # 计算各类别统计
        categories = ['Low', 'Medium', 'High']
        category_stats = {}

        print(f"\n各类别成本下降率统计:")
        print(f"{'类别':<8} {'数量':<6} {'均值%':<8} {'95%CI下限':<10} {'95%CI上限':<10} {'标准差':<8}")
        print("-" * 60)

        for cat in categories:
            cat_data = lng_islands[lng_islands['lng_category'] == cat][target_variable]
            if len(cat_data) > 1:
                # 95%置信区间
                ci_lower, ci_upper = stats.t.interval(0.95, len(cat_data)-1,
                                                     loc=cat_data.mean(),
                                                     scale=stats.sem(cat_data))

                category_stats[cat] = {
                    'count': len(cat_data),
                    'mean': cat_data.mean(),
                    'std': cat_data.std(),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }

                print(f"{cat:<8} {len(cat_data):<6} {cat_data.mean():<8.2f} {ci_lower:<10.2f} {ci_upper:<10.2f} {cat_data.std():<8.2f}")

        # ANOVA检验
        if len(category_stats) >= 2:
            anova_groups = []
            for cat in categories:
                if cat in category_stats:
                    cat_values = lng_islands[lng_islands['lng_category'] == cat][target_variable]
                    anova_groups.append(cat_values)

            if len(anova_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*anova_groups)

                print(f"\nANOVA检验结果（仅有LNG成本的岛屿）:")
                print(f"F统计量: {f_stat:.4f}")
                print(f"P值: {p_value:.6f}")

                if p_value < 0.05:
                    print("结论: 各类别间存在显著差异")

                    # 找出最佳类别
                    best_cat = max(category_stats.keys(), key=lambda x: category_stats[x]['mean'])
                    best_stats = category_stats[best_cat]

                    print(f"\n最佳类别: {best_cat}")
                    print(f"- 平均成本下降: {best_stats['mean']:.2f}%")
                    print(f"- 95%置信区间: [{best_stats['ci_lower']:.2f}%, {best_stats['ci_upper']:.2f}%]")
                    print(f"- 样本数: {best_stats['count']}")
                else:
                    print("结论: 各类别间无显著差异")

        # 基于极端值的特殊分析
        if high_cost_count >= 5:  # 如果有足够的高成本岛屿
            print(f"\n=== 基于极端值的特殊分析 ===")

            # 分为常规成本组和高成本组（使用选定的阈值）
            normal_cost = lng_islands[lng_islands[var_name] < selected_threshold]
            extreme_cost = lng_islands[lng_islands[var_name] >= selected_threshold]

            print(f"常规成本组统计:")
            print(f"  样本数: {len(normal_cost)}")
            print(f"  LNG成本范围: {normal_cost[var_name].min():.1f} - {normal_cost[var_name].max():.1f}")
            print(f"  平均成本下降率: {normal_cost[target_variable].mean():.2f}%")
            print(f"  成本下降率标准差: {normal_cost[target_variable].std():.2f}%")

            print(f"\n高成本组统计:")
            print(f"  样本数: {len(extreme_cost)}")
            print(f"  LNG成本范围: {extreme_cost[var_name].min():.1f} - {extreme_cost[var_name].max():.1f}")
            print(f"  平均成本下降率: {extreme_cost[target_variable].mean():.2f}%")
            print(f"  成本下降率标准差: {extreme_cost[target_variable].std():.2f}%")

            # 统计检验：比较两组成本下降率
            from scipy.stats import ttest_ind
            t_stat, p_val_ttest = ttest_ind(normal_cost[target_variable], extreme_cost[target_variable])

            print(f"\n两组成本下降率比较（t检验）:")
            print(f"  t统计量: {t_stat:.4f}")
            print(f"  p值: {p_val_ttest:.4f}")

            if p_val_ttest < 0.05:
                better_group = "高成本组" if extreme_cost[target_variable].mean() > normal_cost[target_variable].mean() else "常规成本组"
                print(f"  结论: 两组存在显著差异，{better_group}成本下降效果更好")
            else:
                print(f"  结论: 两组成本下降率无显著差异")

            # 效应量计算
            pooled_std = ((normal_cost[target_variable].std()**2 * (len(normal_cost)-1) +
                          extreme_cost[target_variable].std()**2 * (len(extreme_cost)-1)) /
                         (len(normal_cost) + len(extreme_cost) - 2))**0.5

            effect_size = abs(extreme_cost[target_variable].mean() - normal_cost[target_variable].mean()) / pooled_std

            print(f"  效应量(Cohen's d): {effect_size:.3f}")
            if effect_size < 0.2:
                effect_desc = "微小效应"
            elif effect_size < 0.5:
                effect_desc = "小效应"
            elif effect_size < 0.8:
                effect_desc = "中等效应"
            else:
                effect_desc = "大效应"
            print(f"  效应大小: {effect_desc}")
        else:
            print(f"\n高成本岛屿数量过少（{high_cost_count}），跳过极端值特殊分析")
    else:
        print(f"\n样本数较少（{lng_count}），仅进行简单统计分析")

        # 简单的高低分组
        median_lng = lng_islands[var_name].median()
        high_lng = lng_islands[lng_islands[var_name] >= median_lng]
        low_lng = lng_islands[lng_islands[var_name] < median_lng]

        print(f"\n按中位数分组分析:")
        print(f"高{var_desc}组 (≥{median_lng:.3f}): {len(high_lng)}个岛屿，平均成本下降: {high_lng[target_variable].mean():.2f}%")
        print(f"低{var_desc}组 (<{median_lng:.3f}): {len(low_lng)}个岛屿，平均成本下降: {low_lng[target_variable].mean():.2f}%")

    print(f"\n{'='*50}")
    print("LNG专门分析完成")
    print(f"{'='*50}")

# 严谨的分类分析：基于三分位数的成本下降分析
def analyze_cost_reduction_by_categories(data):
    """严谨的分类分析，包含统计检验和置信区间"""
    print("\n" + "="*80)
    print("严谨分类分析：基于三分位数的成本下降效果评估")
    print("="*80)

    # 导入统计检验模块
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')

    # 定义六个核心变量
    six_variables = [
        ('renewable_cost_absolute', '可再生能源成本绝对值'),
        ('storage_cost_absolute', '储能成本绝对值'),
        ('lng_cost_absolute', 'LNG成本绝对值'),
        ('renewable_cost_proportion', '可再生能源成本占比'),
        ('storage_cost_proportion', '储能成本占比'),
        ('lng_cost_proportion', 'LNG成本占比')
    ]

    target_variable = 'total_cost_reduction_rate'

    # 存储所有分析结果
    all_results = {}
    statistical_summary = []

    print(f"\n方法说明:")
    print(f"- 将每个变量按33%和67%分位数分为Low、Medium、High三类")
    print(f"- 计算各类别的总成本下降率统计特征")
    print(f"- 使用ANOVA检验类别间差异的显著性")
    print(f"- 计算95%置信区间评估结果可靠性")
    print(f"- 总样本数: {len(data)}")

    for var_name, var_desc in six_variables:
        print(f"\n" + "="*70)
        print(f"变量分析: {var_desc} ({var_name})")
        print("="*70)

        # 计算分位数
        q33 = data[var_name].quantile(0.33)  # 33%分位数
        q67 = data[var_name].quantile(0.67)  # 67%分位数

        # 处理所有值相同的情况（如LNG成本大部分为0）
        if q33 == q67:
            print(f"警告: 该变量67%的值相同 (值={q33:.3f})，无法进行有效分类")
            # 对于LNG这种大部分为0的变量，按是否为0分类
            if q33 == 0:
                data[f'{var_name}_category'] = 'Zero'
                data.loc[data[var_name] > 0, f'{var_name}_category'] = 'Non-Zero'
                categories = ['Zero', 'Non-Zero']
                print(f"改用二分类: Zero (={q33:.3f}) vs Non-Zero (>{q33:.3f})")

                # 对LNG成本进行专门分析：仅分析有LNG成本的岛屿
                if 'lng_cost' in var_name:
                    print(f"\n{'='*50}")
                    print(f"专门分析：仅有LNG成本的岛屿 ({var_desc})")
                    print(f"{'='*50}")
                    analyze_lng_islands_separately(data, var_name, var_desc, target_variable)
            else:
                continue
        else:
            # 创建三分类标签
            data[f'{var_name}_category'] = 'Medium'  # 默认为中等
            data.loc[data[var_name] <= q33, f'{var_name}_category'] = 'Low'     # 低值
            data.loc[data[var_name] >= q67, f'{var_name}_category'] = 'High'    # 高值
            categories = ['Low', 'Medium', 'High']

            print(f"变量范围: {data[var_name].min():.3f} - {data[var_name].max():.3f}")
            print(f"分位数划分: Low ≤ {q33:.3f} < Medium < {q67:.3f} ≤ High")

        # 计算各类别的详细统计
        category_data = {}
        for cat in categories:
            cat_data = data[data[f'{var_name}_category'] == cat][target_variable]
            if len(cat_data) > 0:
                # 计算95%置信区间
                ci_lower, ci_upper = stats.t.interval(0.95, len(cat_data)-1,
                                                     loc=cat_data.mean(),
                                                     scale=stats.sem(cat_data))

                category_data[cat] = {
                    'count': len(cat_data),
                    'mean': cat_data.mean(),
                    'std': cat_data.std(),
                    'median': cat_data.median(),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'min': cat_data.min(),
                    'max': cat_data.max()
                }

        # 显示统计结果表格
        print(f"\n类别统计结果:")
        print(f"{'类别':<8} {'样本数':<6} {'均值':<7} {'95%CI下限':<9} {'95%CI上限':<9} {'标准差':<7} {'中位数':<7} {'最大值':<7}")
        print("-" * 75)

        for cat in categories:
            if cat in category_data:
                stats_data = category_data[cat]
                print(f"{cat:<8} {stats_data['count']:<6.0f} {stats_data['mean']:<7.2f} "
                      f"{stats_data['ci_lower']:<9.2f} {stats_data['ci_upper']:<9.2f} "
                      f"{stats_data['std']:<7.2f} {stats_data['median']:<7.2f} {stats_data['max']:<7.2f}")

        # ANOVA检验（如果有3个或以上类别且每个类别样本数>1）
        valid_categories = [cat for cat in categories if cat in category_data and category_data[cat]['count'] > 1]

        if len(valid_categories) >= 2:
            # 准备ANOVA数据
            anova_groups = []
            for cat in valid_categories:
                cat_values = data[data[f'{var_name}_category'] == cat][target_variable]
                anova_groups.append(cat_values)

            # 执行ANOVA检验
            if len(anova_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*anova_groups)

                print(f"\nANOVA检验结果:")
                print(f"F统计量: {f_stat:.4f}")
                print(f"p值: {p_value:.6f}")

                if p_value < 0.001:
                    significance = "*** (p < 0.001)"
                elif p_value < 0.01:
                    significance = "** (p < 0.01)"
                elif p_value < 0.05:
                    significance = "* (p < 0.05)"
                else:
                    significance = "ns (不显著)"

                print(f"显著性: {significance}")

                # 效应量 (Eta-squared)
                # 计算组间平方和和总平方和
                all_values = data[target_variable]
                grand_mean = all_values.mean()
                ss_between = sum([category_data[cat]['count'] * (category_data[cat]['mean'] - grand_mean)**2
                                for cat in valid_categories])
                ss_total = sum([(x - grand_mean)**2 for x in all_values])
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                print(f"效应量 (eta-squared): {eta_squared:.4f}")

                # 解释效应量大小
                if eta_squared >= 0.14:
                    effect_size = "大效应"
                elif eta_squared >= 0.06:
                    effect_size = "中等效应"
                elif eta_squared >= 0.01:
                    effect_size = "小效应"
                else:
                    effect_size = "可忽略效应"

                print(f"效应大小: {effect_size}")

        # 找出最佳类别
        if category_data:
            best_category = max(category_data.keys(), key=lambda x: category_data[x]['mean'])
            best_stats = category_data[best_category]

            print(f"\n最佳类别: {best_category}")
            print(f"- 平均成本下降率: {best_stats['mean']:.2f}% "
                  f"(95% CI: {best_stats['ci_lower']:.2f}% - {best_stats['ci_upper']:.2f}%)")
            print(f"- 样本数: {best_stats['count']}")
            print(f"- 相对于其他类别的优势: ", end="")

            # 计算相对优势
            other_means = [category_data[cat]['mean'] for cat in category_data.keys() if cat != best_category]
            if other_means:
                avg_other = sum(other_means) / len(other_means)
                advantage = ((best_stats['mean'] - avg_other) / avg_other) * 100
                print(f"{advantage:+.1f}%")
            else:
                print("无法计算")

            # 保存结果
            all_results[var_name] = {
                'variable_desc': var_desc,
                'best_category': best_category,
                'best_mean': best_stats['mean'],
                'best_ci_lower': best_stats['ci_lower'],
                'best_ci_upper': best_stats['ci_upper'],
                'sample_count': best_stats['count'],
                'anova_p': p_value if 'p_value' in locals() else None,
                'eta_squared': eta_squared if 'eta_squared' in locals() else None,
                'all_categories': category_data
            }

    # 综合结果汇总
    print(f"\n" + "="*80)
    print("综合结果汇总：各变量最佳分类的成本下降效果")
    print("="*80)

    print(f"{'变量':<25} {'最佳类别':<8} {'均值%':<7} {'95%CI':<16} {'样本数':<6} {'ANOVA-p':<10} {'效应量':<8}")
    print("-" * 90)

    # 按最佳效果排序
    sorted_vars = sorted(all_results.items(), key=lambda x: x[1]['best_mean'], reverse=True)

    for var_name, result in sorted_vars:
        ci_str = f"[{result['best_ci_lower']:.1f},{result['best_ci_upper']:.1f}]"
        anova_p_str = f"{result['anova_p']:.3f}" if result['anova_p'] is not None else "N/A"
        eta_str = f"{result['eta_squared']:.3f}" if result['eta_squared'] is not None else "N/A"

        print(f"{result['variable_desc']:<25} {result['best_category']:<8} {result['best_mean']:<7.2f} "
              f"{ci_str:<16} {result['sample_count']:<6.0f} {anova_p_str:<10} {eta_str:<8}")

    # 实际应用建议
    print(f"\n" + "="*80)
    print("实际应用建议")
    print("="*80)

    top_3_vars = sorted_vars[:3]
    print(f"基于分析结果，以下因素对成本下降效果最显著：")

    for i, (var_name, result) in enumerate(top_3_vars, 1):
        print(f"\n{i}. {result['variable_desc']} ({result['best_category']}类别)")
        print(f"   - 平均成本下降: {result['best_mean']:.2f}%")
        print(f"   - 置信区间: {result['best_ci_lower']:.2f}% - {result['best_ci_upper']:.2f}%")
        print(f"   - 建议: 优先考虑{result['best_category'].lower()}类别的系统")

    return all_results

if __name__ == "__main__":
    # 执行六变量专门分析
    results = main_six_variable_analysis()
    print("\n全部分析完成！生成的专门图表:")
    print("1. six_variable_scatter_matrix.png - 六变量散点图矩阵")
    print("2. six_variable_correlation_heatmap.png - 六变量相关性热图")
    print("3. six_variable_regression_coefficients.png - 六变量回归系数图")