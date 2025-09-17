import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_island_energy_cost_analysis():
    """
    对岛屿能源系统的成本进行全面的回归分析和多重共线性检验。
    This function performs a comprehensive regression analysis and multicollinearity check for island energy system costs.
    """
    # --- 0. 环境设置 (Environment Setup) ---
    sns.set(style="whitegrid", rc={'figure.dpi': 100})

    # --- 1. 数据加载与预处理 (Data Loading and Preprocessing) ---
    print("开始加载和处理数据 (Starting to load and process data)...")

    cost_summary_path = '../result/island_cost_summary_0.csv'
    if not os.path.exists(cost_summary_path):
        print(f"错误: 成本汇总文件未找到 '{cost_summary_path}'。请检查路径。")
        return
    cost_df = pd.read_csv(cost_summary_path)

    island_data = []

    for idx, island in cost_df.iterrows():
        lat, lon = island['lat'], island['lon']

        # --- 加载需求数据 ---
        demand_file = f'../demand_get/data/get1/demand_{lat}_{lon}.csv'
        if not os.path.exists(demand_file):
            print(f"警告: 未找到岛屿 ({lat}, {lon}) 的需求文件，跳过该岛屿。")
            continue
        demand_df = pd.read_csv(demand_file)
        
        # --- NEW: Calculate Demand Fluctuations ---
        demand_datetime_series = pd.date_range(start='2020-01-01', periods=len(demand_df), freq='3h')
        demand_df['month'] = demand_datetime_series.month

        # Heating Demand Total and Seasonal Variation (CV方法)
        heating_demand_total = demand_df['heating_demand'].sum()
        h_monthly_mean = demand_df.groupby('month')['heating_demand'].mean()
        h_monthly_mean_filtered = h_monthly_mean[h_monthly_mean > 0.01]
        heating_demand_seasonal = h_monthly_mean_filtered.std() / h_monthly_mean_filtered.mean() if len(h_monthly_mean_filtered) >= 2 and h_monthly_mean_filtered.mean() > 0 else 0

        # Cooling Demand Total and Seasonal Variation (CV方法)
        cooling_demand_total = demand_df['cooling_demand'].sum()
        c_monthly_mean = demand_df.groupby('month')['cooling_demand'].mean()
        c_monthly_mean_filtered = c_monthly_mean[c_monthly_mean > 0.01]
        cooling_demand_seasonal = c_monthly_mean_filtered.std() / c_monthly_mean_filtered.mean() if len(c_monthly_mean_filtered) >= 2 and c_monthly_mean_filtered.mean() > 0 else 0
        
        # --- 加载可再生能源输出数据 ---
        output_file = f'../result/output_0/{lat}_{lon}_results.csv'
        if not os.path.exists(output_file):
            print(f"警告: 未找到岛屿 ({lat}, {lon}) 的能源输出文件，跳过该岛屿。")
            continue
        
        output_df = pd.read_csv(output_file)
        num_steps = len(output_df)
        datetime_series = pd.date_range(start='2020-01-01', periods=num_steps, freq='3h')
        output_df['month'] = datetime_series.month
        output_df['dayofyear'] = datetime_series.dayofyear # For daily fluctuation calculation

        # 计算各可再生能源的总量和季节性/日度变化
        renewables = {'wind': 'WT', 'pv': 'PV', 'wave': 'WEC'}
        renewable_stats = {}

        for name, col in renewables.items():
            if col in output_df.columns and not output_df[col].empty:
                total_utilization = output_df[col].sum()

                # Seasonal Variation (变异系数方法 - Coefficient of Variation)
                monthly_mean = output_df.groupby('month')[col].mean()
                monthly_mean_filtered = monthly_mean[monthly_mean > 0.01]  # 过滤掉发电量很小的月份

                if len(monthly_mean_filtered) >= 2:
                    seasonal_variation = monthly_mean_filtered.std() / monthly_mean_filtered.mean()
                else:
                    seasonal_variation = 0

                # Daily Fluctuation (变异系数CV的平均值 - Coefficient of Variation)
                daily_mean = output_df.groupby('dayofyear')[col].mean()
                daily_std = output_df.groupby('dayofyear')[col].std()
                # 避免除零错误，只计算平均值大于0.1的天数的变异系数
                valid_days = daily_mean > 0.1
                daily_cv = daily_std[valid_days] / daily_mean[valid_days]
                daily_fluctuation = daily_cv.mean() if not daily_cv.empty else 0
                
            else:
                total_utilization, seasonal_variation, daily_fluctuation = 0, 0, 0

            renewable_stats[f'{name}_total'] = total_utilization
            renewable_stats[f'{name}_seasonal'] = seasonal_variation  # 变异系数方法
            renewable_stats[f'{name}_daily'] = daily_fluctuation

        # 汇总该岛屿的所有数据
        island_data.append({
            'lat': lat, 'lon': lon,
            'Heating Demand': heating_demand_total, 'Heating Demand Variation': heating_demand_seasonal,
            'Cooling Demand': cooling_demand_total, 'Cooling Demand Variation': cooling_demand_seasonal,
            'Total Wind Utilization': renewable_stats['wind_total'], 'Wind Seasonal Variation': renewable_stats['wind_seasonal'], 'Wind Daily Fluctuation': renewable_stats['wind_daily'],
            'Total PV Utilization': renewable_stats['pv_total'], 'PV Seasonal Variation': renewable_stats['pv_seasonal'], 'PV Daily Fluctuation': renewable_stats['pv_daily'],
            'Total Wave Utilization': renewable_stats['wave_total'], 'Wave Seasonal Variation': renewable_stats['wave_seasonal'], 'Wave Daily Fluctuation': renewable_stats['wave_daily'],
            'Renewable Cost': island['renewable_cost_per_capita'], 'Storage Cost': island['storage_cost_per_capita'],
            'Electrical Storage Cost': island['electrical_storage_cost_per_capita'], 'Thermal Storage Cost': island['thermal_storage_cost_per_capita'],
            'LNG Cost': island['lng_cost_per_capita'], 'Total Cost': island['renewable_cost_per_capita']+island['storage_cost_per_capita']+island['lng_cost_per_capita']+island['other_equipment_cost_per_capita']+island['discard_cost_per_capita']+island['load_shedding_cost_per_capita']
        })
    
    if not island_data:
        print("错误: 未能处理任何岛屿数据。")
        return
        
    final_df = pd.DataFrame(island_data).fillna(0)
    print("数据处理完成 (Data processing complete).")

    # 打印可再生能源相关变量的描述性统计
    print("\n=== 波浪能变量描述性统计 (Wave Energy Variables Statistics) ===")
    wave_columns = ['Total Wave Utilization', 'Wave Seasonal Variation', 'Wave Daily Fluctuation']
    for col in wave_columns:
        if col in final_df.columns:
            print(f"\n{col}:")
            print(final_df[col].describe())
        else:
            print(f"\n{col}: 列不存在")
    print("=" * 60)

    print("\n=== 光伏变量描述性统计 (PV Energy Variables Statistics) ===")
    pv_columns = ['Total PV Utilization', 'PV Seasonal Variation', 'PV Daily Fluctuation']
    for col in pv_columns:
        if col in final_df.columns:
            print(f"\n{col}:")
            print(final_df[col].describe())
        else:
            print(f"\n{col}: 列不存在")
    print("=" * 60)
    
    print("\n=== 风能变量描述性统计 (Wind Energy Variables Statistics) ===")
    wind_columns = ['Total Wind Utilization', 'Wind Seasonal Variation', 'Wind Daily Fluctuation']
    for col in wind_columns:
        if col in final_df.columns:
            print(f"\n{col}:")
            print(final_df[col].describe())
        else:
            print(f"\n{col}: 列不存在")
    print("=" * 60)
    
    
    # --- 2. 回归分析 (Regression Analysis) ---
    print("正在进行回归分析 (Performing regression analysis)...")
    
    independent_vars = [
        'Heating Demand', 'Heating Demand Variation',
        'Cooling Demand', 'Cooling Demand Variation',
        'Total Wind Utilization', 'Wind Seasonal Variation', 'Wind Daily Fluctuation',
        'Total PV Utilization', 'PV Seasonal Variation', 'PV Daily Fluctuation',
        'Total Wave Utilization', 'Wave Seasonal Variation', 'Wave Daily Fluctuation'
    ]
    dependent_vars = ['Renewable Cost', 'Storage Cost', 'Electrical Storage Cost', 'Thermal Storage Cost', 'LNG Cost', 'Total Cost']
    
    independent_vars = [v for v in independent_vars if v in final_df.columns and final_df[v].nunique() > 1]
    
    variability_vars = [
        'Heating Demand Variation', 'Cooling Demand Variation',
        'Wind Seasonal Variation', 'PV Seasonal Variation', 'Wave Seasonal Variation'
    ]

    # --- 图 1: 回归分析子图 (Plot 1: Regression Subplots) ---
    print("正在生成图1：回归分析子图 (Generating Plot 1: Regression Subplots)...")
    n_rows, n_cols = len(dependent_vars), len(independent_vars)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.5))
    fig.suptitle('Simple Linear Regression: Predictors vs. Costs', fontsize=22, y=1.02)

    regression_results = {}
    for i, y_var in enumerate(dependent_vars):
        for j, x_var in enumerate(independent_vars):
            ax = axes[i, j]
            
            if x_var in variability_vars:
                # 针对不同可再生能源类型使用不同的筛选策略
                if 'Wave' in x_var:
                    # 波浪能：需要有显著的总利用量和变异性才纳入分析
                    wave_utilization_condition = final_df['Total Wave Utilization'] > 100  # 有一定的波浪能利用
                    wave_variability_condition = final_df[x_var] > 0.2  # 变异系数阈值
                    current_df = final_df[wave_utilization_condition & wave_variability_condition].copy()
                    min_threshold_text = 'Wave utilization>100\n& variability>0.2'
                else:
                    # 其他可再生能源：使用变异系数阈值
                    current_df = final_df[final_df[x_var] > 0.1].copy()
                    min_threshold_text = 'variability>0.1'

                if len(current_df) < 2:
                    ax.text(0.5, 0.5, f'Not enough data\nwith {min_threshold_text}', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
                    regression_results[(y_var, x_var)] = {'r2': np.nan, 'p_value': np.nan}
                    ax.set_xlabel(x_var, fontsize=9)
                    ax.set_ylabel(y_var if j == 0 else "", fontsize=11)
                    continue
            else:
                current_df = final_df.copy()
            
            X = sm.add_constant(current_df[x_var])
            y = current_df[y_var]
            model = sm.OLS(y, X).fit()
            
            regression_results[(y_var, x_var)] = {'r2': model.rsquared, 'p_value': model.pvalues[x_var]}
            
            sns.regplot(x=x_var, y=y_var, data=current_df, ax=ax, ci=None,
                        line_kws={'color': 'red', 'linestyle': '--'},
                        scatter_kws={'alpha': 0.6, 's': 40})
            
            p_val = model.pvalues[x_var]
            p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            ax.text(0.05, 0.95, f'$R^2 = {model.rsquared:.2f}$\n{p_text}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            ax.set_xlabel(x_var, fontsize=9)
            ax.set_ylabel(y_var if j == 0 else "", fontsize=11)
            ax.tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('regression/regression_subplots1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("图1已保存为 regression_subplots.png")

    # --- 图 2: R方和P值矩阵图 (Plot 2: R-squared and P-value Matrix) ---
    print("正在生成图2：R方和P值矩阵图 (Generating Plot 2: R-squared and P-value Matrix)...")
    r2_matrix = pd.DataFrame(index=dependent_vars, columns=independent_vars, dtype=float)
    
    for (y_var, x_var), result in regression_results.items():
        r2_matrix.loc[y_var, x_var] = result['r2']

    annotations = r2_matrix.copy().astype(str)
    for (y_var, x_var), result in regression_results.items():
         if pd.isna(result['p_value']):
             annotations.loc[y_var, x_var] = 'N/A'
             continue
         p_val = result['p_value']
         p_text = f"p < 0.001" if p_val < 0.001 else f"p={p_val:.3f}"
         annotations.loc[y_var, x_var] = f'R²={result["r2"]:.2f}\n({p_text})'

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(r2_matrix, annot=annotations, fmt='', cmap="viridis", ax=ax,
                annot_kws={"size": 9, "color": "white"}, cbar_kws={'label': 'R-squared (R²)'})
    ax.set_title('Matrix of R-squared and P-values', fontsize=18)
    
    # --- FIX START ---
    # The keyword for font size in setp is 'size', not 'labelsize'.
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", size=11)
    # --- FIX END ---

    plt.tight_layout()
    plt.savefig('regression/r2_pvalue_matrix1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("图2已保存为 r2_pvalue_matrix.png")
    
    # --- 3. 多重共线性检验 (Multicollinearity Test) ---
    print("正在进行多重共线性检验 (Performing multicollinearity test)...")
    X_multi = final_df[independent_vars].dropna()

    # --- 图 3: 多重共线性检验图 (Plot 3: Multicollinearity Diagnostics) ---
    print("正在生成图3：多重共线性检验图 (Generating Plot 3: Multicollinearity Diagnostics)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle('Multicollinearity Diagnostics for Independent Variables', fontsize=20)

    # Subplot 1: Correlation Matrix
    corr_matrix = X_multi.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax1,
                annot_kws={"size": 9})
    ax1.set_title('Correlation Matrix', fontsize=16)

    # --- FIX START ---
    # Apply the same fix here: use 'size' instead of 'labelsize'.
    ax1.tick_params(axis='y', rotation=0)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", size=11)
    # --- FIX END ---

    # Subplot 2: Variance Inflation Factor (VIF)
    X_vif = sm.add_constant(X_multi)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    vif_data = vif_data[vif_data["feature"] != "const"].sort_values('VIF', ascending=False)
    
    sns.barplot(x='VIF', y='feature', data=vif_data, ax=ax2, palette='mako', hue='feature', legend=False)
    ax2.set_title('Variance Inflation Factor (VIF)', fontsize=16)
    ax2.axvline(x=5, color='orange', linestyle='--', label='Moderate Concern (VIF=5)')
    ax2.axvline(x=10, color='red', linestyle='--', label='High Concern (VIF=10)')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('regression/multicollinearity_analysis1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("图3已保存为 multicollinearity_analysis.png")
    print("\n所有分析和绘图已完成 (All analysis and plotting complete)!")

# 运行主函数
if __name__ == '__main__':
    run_island_energy_cost_analysis()