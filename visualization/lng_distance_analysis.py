#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岛屿LNG成本与距离LNG港口距离关系分析
Analysis of Island LNG Cost vs Distance to LNG Ports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from geopy.distance import great_circle
import geopandas as gpd
from shapely.geometry import Point
import warnings

warnings.filterwarnings('ignore')

# 设置字体和样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

def load_ipcc_regions():
    """加载IPCC区域数据"""
    print("🗺️  Loading IPCC regions...")
    
    try:
        # 读取IPCC区域GeoJSON文件
        ipcc_regions = gpd.read_file('IPCC-WGI-reference-regions-v4.geojson')
        print(f"✓ Loaded {len(ipcc_regions)} IPCC regions")
        
        return ipcc_regions
        
    except FileNotFoundError:
        print("⚠️  Warning: Cannot find IPCC-WGI-reference-regions-v4.geojson file")
        print("Will skip IPCC region mapping")
        return None
    except Exception as e:
        print(f"❌ Error reading IPCC regions file: {e}")
        return None

def load_island_lng_data():
    """Load island and LNG terminal data"""
    print("🏝️  Loading island data...")
    # Read island data
    islands = pd.read_csv("filtered_island_1898.csv")
    # Clean column names and extract coordinates
    islands.columns = [col.strip() for col in islands.columns]
    if 'Long' in islands.columns and 'Lat' in islands.columns:
        islands['lon'] = islands['Long']
        islands['lat'] = islands['Lat']
    print(f"✓ Loaded {len(islands)} islands")
    
    # Load cost data and merge
    print("💰 Loading cost data...")
    cost_file = 'island_cost_summary_0.csv'
    cost_df = pd.read_csv(cost_file)
    
    # Merge island coordinates with cost data
    merged_df = pd.merge(islands, cost_df, on=['lat', 'lon'], how='inner')
    print(f"✓ Merged data for {len(merged_df)} islands with cost information")
    
    print("⛽ Loading LNG terminal data...")
    # Read LNG terminal data
    lng_terminals = pd.read_excel("LNG_Terminals.xlsx")
    # Filter only operational terminals with valid coordinates
    lng_active = lng_terminals[
        (lng_terminals['Status'].isin(['Operating', 'Under Construction'])) &
        (lng_terminals['Latitude'].notna()) & 
        (lng_terminals['Longitude'].notna())
    ].copy()
    lng_active['lon'] = lng_active['Longitude'] 
    lng_active['lat'] = lng_active['Latitude']
    print(f"✓ Loaded {len(lng_active)} active LNG terminals")
    
    return merged_df, lng_active

def map_islands_to_ipcc_regions(df, ipcc_regions):
    """将岛屿坐标映射到IPCC区域"""
    if ipcc_regions is None:
        print("⚠️  No IPCC region data, skipping mapping")
        return df
    
    print("🗺️  Mapping islands to IPCC regions...")
    
    # 为岛屿创建Point几何对象
    island_points = [Point(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
    
    ipcc_codes = []
    ipcc_names = []
    continents = []
    
    for i, point in enumerate(island_points):
        # 查找包含该点的IPCC区域
        region_found = False
        
        for _, region in ipcc_regions.iterrows():
            try:
                if region.geometry.contains(point):
                    # 尝试获取区域信息
                    code = region.get('Acronym', f'Region_{region.name}')
                    name = region.get('Name', f'Region_{region.name}')
                    continent = region.get('Continent', 'Unknown')
                    
                    ipcc_codes.append(code)
                    ipcc_names.append(name)
                    continents.append(continent)
                    region_found = True
                    break
            except Exception as e:
                continue
        
        if not region_found:
            ipcc_codes.append('Unknown')
            ipcc_names.append('Unknown Region')
            continents.append('Unknown')
    
    # 添加到DataFrame
    df['IPCC_Region_Code'] = ipcc_codes
    df['IPCC_Region_Name'] = ipcc_names
    df['Continent'] = continents
    
    # 统计映射结果
    mapped_count = len(df[df['IPCC_Region_Code'] != 'Unknown'])
    print(f"✓ Successfully mapped {mapped_count} islands to IPCC regions")
    print(f"⚠️  Unmapped islands: {len(df) - mapped_count}")
    
    # 显示区域分布
    print("\n📊 IPCC Region Distribution (Top 10):")
    region_counts = df['IPCC_Region_Code'].value_counts().head(10)
    for region, count in region_counts.items():
        print(f"  {region}: {count} islands")
    
    return df

def calculate_min_distances(islands, lng_terminals):
    """Calculate minimum great circle distance from each island to nearest LNG terminal"""
    print("📏 Calculating distances to nearest LNG terminals...")
    
    min_distances = []
    nearest_ports = []
    
    for idx, island in islands.iterrows():
        island_coord = (island['lat'], island['lon'])
        
        # Calculate distance to all LNG terminals
        distances = []
        for _, terminal in lng_terminals.iterrows():
            terminal_coord = (terminal['lat'], terminal['lon'])
            # Calculate great circle distance in km and multiply by 1.2
            distance = great_circle(island_coord, terminal_coord).kilometers * 1.2
            distances.append(distance)
        
        # Find minimum distance and corresponding terminal
        if distances:
            min_distance = min(distances)
            min_idx = distances.index(min_distance)
            nearest_terminal = lng_terminals.iloc[min_idx]
            
            # Try to get terminal name
            port_name = nearest_terminal.get('Name', f"Terminal_{min_idx}")
        else:
            min_distance = np.nan
            port_name = None
        
        min_distances.append(min_distance)
        nearest_ports.append(port_name)
    
    islands['distance_to_lng_port'] = min_distances
    islands['nearest_lng_port'] = nearest_ports
    
    valid_distances = len(islands[~pd.isna(islands['distance_to_lng_port'])])
    print(f"✓ Calculated distances for {valid_distances} islands")
    
    return islands

def create_lng_distance_analysis(df):
    """创建LNG成本与距离关系的综合分析图"""
    
    # 筛选有LNG成本和距离数据的岛屿
    analysis_data = df[(df['lng_cost_per_capita'] > 0) & 
                      (~pd.isna(df['distance_to_lng_port']))].copy()
    
    print(f"分析数据包含 {len(analysis_data)} 个岛屿")
    
    if len(analysis_data) < 10:
        print("数据量太少，无法进行有效分析")
        return None
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('LNG Cost vs Distance to LNG Ports Analysis', fontsize=16, fontweight='bold')
    
    # 子图1: 散点图 + 回归线
    ax1 = axes[0, 0]
    scatter = ax1.scatter(analysis_data['distance_to_lng_port'], 
                         analysis_data['lng_cost_per_capita'],
                         c=analysis_data['population'], 
                         cmap='viridis', 
                         alpha=0.7, 
                         s=50)
    
    # 计算回归线
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        analysis_data['distance_to_lng_port'], 
        analysis_data['lng_cost_per_capita']
    )
    
    x_reg = np.linspace(analysis_data['distance_to_lng_port'].min(), 
                       analysis_data['distance_to_lng_port'].max(), 100)
    y_reg = slope * x_reg + intercept
    
    ax1.plot(x_reg, y_reg, 'r-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Distance to Nearest LNG Port (km)')
    ax1.set_ylabel('LNG Cost per Capita ($)')
    ax1.set_title('LNG Cost vs Distance (colored by population)')
    
    # 添加R²值
    ax1.text(0.05, 0.95, f'R² = {r_value**2:.3f}\np = {p_value:.2e}', 
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加颜色条
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Population', fontsize=9)
    
    # 子图2: 按距离区间分组的箱线图
    ax2 = axes[0, 1]
    
    # 创建距离区间
    analysis_data['distance_bins'] = pd.cut(analysis_data['distance_to_lng_port'], 
                                           bins=5, labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    box_data = [analysis_data[analysis_data['distance_bins'] == bin_label]['lng_cost_per_capita'].values 
                for bin_label in analysis_data['distance_bins'].cat.categories if len(analysis_data[analysis_data['distance_bins'] == bin_label]) > 0]
    
    labels = [label for label in analysis_data['distance_bins'].cat.categories 
              if len(analysis_data[analysis_data['distance_bins'] == label]) > 0]
    
    if box_data:
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        
        # 设置箱线图颜色
        colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red']
        for i, patch in enumerate(bp['boxes']):
            if i < len(colors):
                patch.set_facecolor(colors[i])
    
    ax2.set_xlabel('Distance Percentile Groups')
    ax2.set_ylabel('LNG Cost per Capita ($)')
    ax2.set_title('LNG Cost Distribution by Distance Groups')
    ax2.tick_params(axis='x', rotation=45)
    
    # 子图3: 按IPCC区域分类的散点图
    ax3 = axes[1, 0]
    
    # 使用IPCC区域分组
    if 'IPCC_Region_Code' in analysis_data.columns:
        ipcc_regions = analysis_data['IPCC_Region_Code'].unique()
        # 只显示数据点数量最多的前8个IPCC区域，避免图例过于拥挤
        region_counts = analysis_data['IPCC_Region_Code'].value_counts()
        top_regions = region_counts.head(8).index
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(top_regions)))
        
        for i, region in enumerate(top_regions):
            region_data = analysis_data[analysis_data['IPCC_Region_Code'] == region]
            if len(region_data) > 0:
                ax3.scatter(region_data['distance_to_lng_port'], 
                           region_data['lng_cost_per_capita'],
                           c=[colors[i]], label=f'{region} (n={len(region_data)})', 
                           alpha=0.7, s=50)
        
        # 其余区域用灰色显示
        other_data = analysis_data[~analysis_data['IPCC_Region_Code'].isin(top_regions)]
        if len(other_data) > 0:
            ax3.scatter(other_data['distance_to_lng_port'], 
                       other_data['lng_cost_per_capita'],
                       c='gray', label=f'Others (n={len(other_data)})', 
                       alpha=0.5, s=30)
        
        ax3.set_title('LNG Cost vs Distance by IPCC Region')
    else:
        # 如果没有IPCC区域数据，按人口规模分组
        analysis_data['population_group'] = pd.cut(analysis_data['population'], 
                                                 bins=[0, 100, 500, 1000, float('inf')],
                                                 labels=['Small (<100)', 'Medium (100-500)', 
                                                       'Large (500-1000)', 'Very Large (>1000)'])
        
        pop_groups = analysis_data['population_group'].cat.categories
        colors = plt.cm.Set1(np.linspace(0, 1, len(pop_groups)))
        
        for i, pop_group in enumerate(pop_groups):
            group_data = analysis_data[analysis_data['population_group'] == pop_group]
            if len(group_data) > 0:
                ax3.scatter(group_data['distance_to_lng_port'], 
                           group_data['lng_cost_per_capita'],
                           c=[colors[i]], label=pop_group, alpha=0.7, s=50)
        
        ax3.set_title('LNG Cost vs Distance by Population Group')
    
    ax3.set_xlabel('Distance to Nearest LNG Port (km)')
    ax3.set_ylabel('LNG Cost per Capita ($)')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 子图4: 距离vs成本的密度热图
    ax4 = axes[1, 1]
    
    # 创建2D直方图
    hist, x_edges, y_edges = np.histogram2d(analysis_data['distance_to_lng_port'], 
                                           analysis_data['lng_cost_per_capita'], 
                                           bins=20)
    
    # 创建网格
    X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
    
    # 绘制热图
    im = ax4.contourf(X, Y, hist.T, levels=10, cmap='YlOrRd', alpha=0.8)
    
    # 叠加散点图
    ax4.scatter(analysis_data['distance_to_lng_port'], 
               analysis_data['lng_cost_per_capita'],
               c='black', alpha=0.4, s=20)
    
    ax4.set_xlabel('Distance to Nearest LNG Port (km)')
    ax4.set_ylabel('LNG Cost per Capita ($)')
    ax4.set_title('Density Distribution')
    
    # 添加颜色条
    cbar2 = plt.colorbar(im, ax=ax4)
    cbar2.set_label('Island Count', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig, analysis_data, (slope, intercept, r_value, p_value)

def print_statistical_analysis(analysis_data, regression_stats):
    """打印详细的统计分析结果"""
    slope, intercept, r_value, p_value = regression_stats
    
    print("\n" + "="*60)
    print("LNG成本与距离关系统计分析")
    print("="*60)
    
    print(f"\n基本统计:")
    print(f"  分析样本数: {len(analysis_data)}")
    print(f"  距离范围: {analysis_data['distance_to_lng_port'].min():.1f} - {analysis_data['distance_to_lng_port'].max():.1f} km")
    print(f"  LNG成本范围: ${analysis_data['lng_cost_per_capita'].min():.2f} - ${analysis_data['lng_cost_per_capita'].max():.2f}")
    
    print(f"\n回归分析:")
    print(f"  回归方程: LNG成本 = {slope:.4f} × 距离 + {intercept:.2f}")
    print(f"  相关系数 R: {r_value:.3f}")
    print(f"  决定系数 R²: {r_value**2:.3f}")
    print(f"  显著性 p值: {p_value:.2e}")
    
    significance = "显著" if p_value < 0.05 else "不显著"
    print(f"  统计显著性: {significance}")
    
    # 相关性强度判断
    r_abs = abs(r_value)
    if r_abs >= 0.7:
        correlation_strength = "强"
    elif r_abs >= 0.3:
        correlation_strength = "中等"
    else:
        correlation_strength = "弱"
    
    correlation_direction = "正" if r_value > 0 else "负"
    print(f"  相关性强度: {correlation_strength}{correlation_direction}相关")
    
    # 按IPCC区域统计
    if 'IPCC_Region_Code' in analysis_data.columns:
        print(f"\n各IPCC区域统计 (前10个):")
        region_stats = analysis_data.groupby('IPCC_Region_Code').agg({
            'distance_to_lng_port': ['count', 'mean', 'std'],
            'lng_cost_per_capita': ['mean', 'std']
        }).round(2)
        
        # 按岛屿数量排序，显示前10个区域
        region_stats_sorted = region_stats.sort_values(('distance_to_lng_port', 'count'), ascending=False)
        
        for region in region_stats_sorted.head(10).index:
            count = region_stats_sorted.loc[region, ('distance_to_lng_port', 'count')]
            avg_dist = region_stats_sorted.loc[region, ('distance_to_lng_port', 'mean')]
            avg_cost = region_stats_sorted.loc[region, ('lng_cost_per_capita', 'mean')]
            print(f"  {region}: {count}个岛屿, 平均距离{avg_dist:.0f}km, 平均成本${avg_cost:.2f}")
    
    # 距离分组分析
    print(f"\n距离分组分析:")
    analysis_data['distance_quartiles'] = pd.qcut(analysis_data['distance_to_lng_port'], 
                                                  q=4, labels=['Q1(近)', 'Q2', 'Q3', 'Q4(远)'])
    
    quartile_stats = analysis_data.groupby('distance_quartiles').agg({
        'lng_cost_per_capita': ['count', 'mean', 'std'],
        'distance_to_lng_port': ['mean']
    }).round(2)
    
    for quartile in quartile_stats.index:
        count = quartile_stats.loc[quartile, ('lng_cost_per_capita', 'count')]
        avg_cost = quartile_stats.loc[quartile, ('lng_cost_per_capita', 'mean')]
        avg_dist = quartile_stats.loc[quartile, ('distance_to_lng_port', 'mean')]
        print(f"  {quartile}: {count}个岛屿, 平均距离{avg_dist:.0f}km, 平均成本${avg_cost:.2f}")
    
    # 最近港口统计
    print(f"\n最常用的LNG港口 (前10个):")
    port_counts = analysis_data['nearest_lng_port'].value_counts().head(10)
    for port, count in port_counts.items():
        port_data = analysis_data[analysis_data['nearest_lng_port'] == port]
        avg_dist = port_data['distance_to_lng_port'].mean()
        avg_cost = port_data['lng_cost_per_capita'].mean()
        print(f"  {port}: {count}个岛屿, 平均距离{avg_dist:.0f}km, 平均成本${avg_cost:.2f}")

def main():
    """主函数"""
    print("🚀 Starting LNG Cost vs Distance Analysis...")
    
    # Load island and LNG data
    islands, lng_terminals = load_island_lng_data()
    
    # Load IPCC regions and map islands
    ipcc_regions = load_ipcc_regions()
    islands = map_islands_to_ipcc_regions(islands, ipcc_regions)
    
    # Calculate distances to nearest LNG terminals
    df_with_distances = calculate_min_distances(islands, lng_terminals)
    
    # 创建分析图表
    result = create_lng_distance_analysis(df_with_distances)
    
    if result:
        fig, analysis_data, regression_stats = result
        
        # 打印统计分析
        print_statistical_analysis(analysis_data, regression_stats)
        
        print("\n分析完成！")
    
    else:
        print("分析失败，请检查数据")

if __name__ == "__main__":
    main()