import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats


def set_axis_scientific(ax):
      ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
      ax.xaxis.offsetText.set_fontsize(20)  # 增大科学计数法字体
      ax.xaxis.offsetText.set_fontfamily('Arial')
      ax.yaxis.offsetText.set_fontsize(20)  # 增大科学计数法字体
      ax.yaxis.offsetText.set_fontfamily('Arial')
      
def run_island_energy_cost_analysis():
    """
    对岛屿能源系统的成本进行全面的回归分析和多重共线性检验。
    This function performs a comprehensive regression analysis and multicollinearity check for island energy system costs.
    """
    # --- 0. 环境设置 (Environment Setup) ---
    sns.set(style="white", rc={'figure.dpi': 300})  # 使用white样式避免网格线

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

    # --- 辅助函数 (Helper Functions) ---
    def get_significance_stars(p_value):
        """根据p值返回显著性星号"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

    # --- 图 4: 竖排三个散点回归图 (Three Vertical Scatter Regression Plots) ---
    print("正在生成竖排三个散点回归图 (Generating three vertical scatter regression plots)...")

    # 准备数据，筛选有效数据
    plot_data = final_df.copy()

    # 检查数据可用性
    wave_storage_data = plot_data[(plot_data['Total Wave Utilization'] > 100) &
                                  (plot_data['Wave Seasonal Variation'] > 0.2)].copy()
    wind_lng_data = plot_data[plot_data['Wind Seasonal Variation'] > 0.1].copy()
    wind_total_data = plot_data[plot_data['Wind Seasonal Variation'] > 0.1].copy()

    # 检查是否有足够的数据
    data_available = [len(wave_storage_data) >= 2, len(wind_lng_data) >= 2, len(wind_total_data) >= 2]

    if any(data_available):
        # 创建2x1的子图布局，第一个是波浪能图，第二个是风能双y轴图
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # 调整为2个子图
        fig.subplots_adjust(hspace=0.4)  # 增加子图间距以避免重叠

        # 第一个图：波浪能季节变异性 vs 储能成本
        ax1 = axes[0]
        if data_available[0]:
            x_data = wave_storage_data['Wave Seasonal Variation']
            y_data = wave_storage_data['Storage Cost']
            color = '#2E8B57'  # 海绿色

            # 散点图 - 更大的点
            ax1.scatter(x_data, y_data, s=20, c=color, alpha=0.7, edgecolors='none')

            # 回归分析
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)

            # 回归线
            x_reg = np.linspace(x_data.min(), x_data.max(), 100)
            y_reg = slope * x_reg + intercept
            ax1.plot(x_reg, y_reg, color='black', linewidth=2, alpha=0.8)

            # 获取显著性星号
            stars = get_significance_stars(p_value)

            # 添加R²和显著性星号的注释
            ax1.text(0.05, 0.95, f'R² = {r_value**2:.3f}{stars}', transform=ax1.transAxes,
                    fontsize=24, fontfamily='Arial', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='none'))

            # 设置科学计数法
            set_axis_scientific(ax1)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax1.transAxes, fontsize=24, color='gray', fontfamily='Arial')

        # 设置第一个图的标签和格式
        ax1.set_xlabel('Wave seasonal variability', fontsize=28, fontfamily='Arial')  # 增大字体
        ax1.set_ylabel('Storage Cost(USD/per)', fontsize=28, fontfamily='Arial')  # 增大字体
        ax1.tick_params(axis='both', which='major', labelsize=28, width=1, length=4)  # 增大刻度标签字体

        # 设置刻度标签字体
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontfamily('Arial')

        # 移除顶部和右侧边框，确保无网格
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['left'].set_color('black')  # 确保左边框为黑色
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['bottom'].set_color('black')  # 确保底边框为黑色
        ax1.grid(False)  # 确保无网格线

        # 第二个图：风能季节变异性的双y轴图
        ax2 = axes[1]

        if data_available[1] and data_available[2]:
            # 创建第二个y轴
            ax2_right = ax2.twinx()

            # 左y轴：LNG成本 (蓝色)
            x_data_lng = wind_lng_data['Wind Seasonal Variation']
            y_data_lng = wind_lng_data['LNG Cost']
            color_lng = '#4682B4'  # 钢蓝色

            # 散点图 - LNG成本
            ax2.scatter(x_data_lng, y_data_lng, s=20, c=color_lng, alpha=0.7, edgecolors='none', label='LNG cost')

            # 回归分析 - LNG
            slope_lng, intercept_lng, r_value_lng, p_value_lng, std_err_lng = stats.linregress(x_data_lng, y_data_lng)
            x_reg_lng = np.linspace(x_data_lng.min(), x_data_lng.max(), 100)
            y_reg_lng = slope_lng * x_reg_lng + intercept_lng
            ax2.plot(x_reg_lng, y_reg_lng, color=color_lng, linewidth=2, alpha=0.8)

            # 右y轴：总成本 (红色)
            x_data_total = wind_total_data['Wind Seasonal Variation']
            y_data_total = wind_total_data['Total Cost']
            color_total = '#DC143C'  # 深红色

            # 散点图 - 总成本
            ax2_right.scatter(x_data_total, y_data_total, s=20, c=color_total, alpha=0.7, edgecolors='none', label='Total cost')

            # 回归分析 - 总成本
            slope_total, intercept_total, r_value_total, p_value_total, std_err_total = stats.linregress(x_data_total, y_data_total)
            x_reg_total = np.linspace(x_data_total.min(), x_data_total.max(), 100)
            y_reg_total = slope_total * x_reg_total + intercept_total
            ax2_right.plot(x_reg_total, y_reg_total, color=color_total, linewidth=2, alpha=0.8)

            # 获取显著性星号
            stars_lng = get_significance_stars(p_value_lng)
            stars_total = get_significance_stars(p_value_total)

            # 添加R²注释 - LNG (左上角)
            ax2.text(0.05, 0.95, f'LNG: R² = {r_value_lng**2:.3f}{stars_lng}', transform=ax2.transAxes,
                    fontsize=24, fontfamily='Arial', verticalalignment='top', color=color_lng,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

            # 添加R²注释 - 总成本 (右上角)
            ax2.text(0.95, 0.95, f'Total: R² = {r_value_total**2:.3f}{stars_total}', transform=ax2.transAxes,
                    fontsize=24, fontfamily='Arial', verticalalignment='top', horizontalalignment='right', color=color_total,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

            # 设置左y轴标签 - 保持黑色
            ax2.set_ylabel('LNG Cost(USD/per)', fontsize=28, fontfamily='Arial', color='black')  # 坐标轴标签为黑色
            ax2.tick_params(axis='y', labelcolor='black', labelsize=28, width=1, length=4)  # 刻度标签为黑色
            ax2.spines['left'].set_color('black')  # 坐标轴为黑色

            # 设置右y轴标签 - 保持黑色
            ax2_right.set_ylabel('Total Cost(USD/per)', fontsize=28, fontfamily='Arial', color='black')  # 坐标轴标签为黑色
            ax2_right.tick_params(axis='y', labelcolor='black', labelsize=28, width=1, length=4)  # 刻度标签为黑色
            ax2_right.spines['right'].set_color('black')  # 坐标轴为黑色

            # 调整右y轴的范围，使数据点在图中显示更高，分离两条回归线
            y_total_min, y_total_max = ax2_right.get_ylim()
            y_total_range = y_total_max - y_total_min
            # 将右y轴的下限向下扩展，使数据相对位置更高
            ax2_right.set_ylim(y_total_min - y_total_range * 0.3, y_total_max)

            # 设置科学计数法
            set_axis_scientific(ax2)
            set_axis_scientific(ax2_right)

        else:
            ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax2.transAxes, fontsize=28, color='gray', fontfamily='Arial')

        # 设置第二个图的x轴标签
        ax2.set_xlabel('Wind seasonal variability', fontsize=28, fontfamily='Arial')  # 增大字体
        ax2.tick_params(axis='x', labelsize=28, width=1, length=4)  # 增大刻度标签字体

        # 设置x轴刻度标签字体
        for label in ax2.get_xticklabels():
            label.set_fontfamily('Arial')

        # 移除顶部和右侧顶部边框，确保无网格
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_linewidth(1)
        ax2.spines['bottom'].set_color('black')  # 确保底边框为黑色
        ax2.grid(False)  # 确保无网格线
        ax2_right.grid(False)  # 确保右y轴也无网格线
        ax2_right.spines['top'].set_visible(False)  # 移除右y轴的顶部边框

        # 对齐y轴标签
        fig.align_ylabels([ax1, ax2])

        plt.tight_layout()
        plt.savefig('regression/combined_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("合并散点回归图已保存为 combined_scatter_plots.png")
    else:
        print("所有图表的数据都不足，无法生成散点回归图")

    print("\n所有分析和绘图已完成 (All analysis and plotting complete)!")

# 运行主函数
if __name__ == '__main__':
    run_island_energy_cost_analysis()
    
