# 9. --- Nature Communications风格四层桑基图：Region -> Ideal -> Baseline -> Climate Stress ---

import plotly.graph_objects as go  # 导入plotly桑基图绘制库
import plotly.io as pio  # 导入plotly IO库用于格式设置
import matplotlib.cm as cm  # matplotlib颜色映射库
import matplotlib.colors as mcolors  # matplotlib颜色工具
import numpy as np  # 数值计算库

print("开始创建Nature Communications风格四层桑基图（仿照output.png样式）...")

# 设置plotly默认样式为静态图，符合期刊要求
pio.renderers.default = "browser"  # 设置输出渲染器为浏览器

# =============================================================================
# 1. 数据检查和预处理
# =============================================================================

# 检查桑基图数据是否存在
if 'sankey_df' not in locals():
    print("错误：sankey_df数据不存在，请先运行上一个cell生成数据")
else:
    print(f"桑基图数据检查通过，包含 {len(sankey_df)} 个岛屿")
    print(f"涉及区域: {sankey_df['region'].nunique()} 个")
    print(f"涉及分类组合: {len(sankey_df[['ideal_class', 'baseline_class', 'climate_class']].drop_duplicates())} 种")

# =============================================================================
# 2. Nature风格颜色映射 - 仿照output.png的简洁配色
# =============================================================================

# 定义四个主要类别的颜色（仿照output.png）
category_colors = {
    'High Affordable\\nLow Cost': 'rgba(255, 223, 0, 0.85)',      # 金黄色，对应output.png底部
    'High Affordable\\nHigh Cost': 'rgba(255, 127, 127, 0.85)',   # 粉红色，对应output.png中部偏下
    'Low Affordable\\nLow Cost': 'rgba(64, 224, 224, 0.85)',      # 青色，对应output.png主体
    'Low Affordable\\nHigh Cost': 'rgba(255, 140, 0, 0.85)'       # 橙色，对应output.png上部
}

print(f"使用Nature期刊风格的四色配色方案")

# 为区域生成基于类别主色调的变化色彩
def generate_region_colors_from_categories(regions, base_colors):
    """基于四个主要类别颜色生成区域颜色变化"""
    region_colors = {}
    base_color_list = list(base_colors.values())
    n_regions = len(regions)
    
    for i, region in enumerate(regions):
        # 循环使用四种基础颜色，并添加微调
        base_idx = i % 4
        base_color = base_color_list[base_idx]
        
        # 提取RGB值并进行微调
        rgb_match = base_color.replace('rgba(', '').replace(')', '').split(', ')
        r, g, b = int(rgb_match[0]), int(rgb_match[1]), int(rgb_match[2])
        
        # 为每个区域添加细微的色彩变化
        variation = (i // 4) * 15  # 每4个区域为一组，组间有色彩变化
        r = min(255, max(0, r + variation - 30))
        g = min(255, max(0, g + variation - 15))  
        b = min(255, max(0, b + variation - 10))
        
        region_colors[region] = f'rgba({r}, {g}, {b}, 0.75)'
    
    return region_colors

# 获取所有唯一区域并生成颜色
unique_regions = sorted(sankey_df['region'].unique())
region_colors = generate_region_colors_from_categories(unique_regions, category_colors)

print(f"为 {len(unique_regions)} 个区域生成了基于类别的颜色变化")

# =============================================================================
# 3. 构建四层桑基图的节点和链接数据结构
# =============================================================================

# 定义四个层级的所有可能节点
all_regions = sorted(sankey_df['region'].unique())  # 第1层：区域
all_ideal_classes = sorted(sankey_df['ideal_class'].unique())  # 第2层：Ideal分类
all_baseline_classes = sorted(sankey_df['baseline_class'].unique())  # 第3层：Baseline分类
all_climate_classes = sorted(sankey_df['climate_class'].unique())  # 第4层：Climate Stress分类

# 创建节点标签列表（按层级顺序）- 仿照output.png的简洁标签
node_labels = []
node_colors = []  # 节点颜色列表

# 第1层：区域节点 - 使用区域代码（仿照output.png左侧标签）
for region in all_regions:
    node_labels.append(region)  # 简洁的区域标签
    node_colors.append(region_colors[region])  # 使用区域对应颜色

# 第2层：Ideal分类节点 - 使用类别颜色
for ideal_class in all_ideal_classes:
    node_labels.append("")  # 不显示中间节点标签，仿照output.png
    node_colors.append(category_colors[ideal_class])  # 使用类别对应颜色

# 第3层：Baseline分类节点 - 使用类别颜色
for baseline_class in all_baseline_classes:
    node_labels.append("")  # 不显示中间节点标签
    node_colors.append(category_colors[baseline_class])

# 第4层：Climate Stress分类节点 - 显示最终类别标签（仿照output.png右侧）
for climate_class in all_climate_classes:
    # 简化标签，仿照output.png的样式
    if climate_class == 'High Affordable\\nLow Cost':
        display_label = "High Affordable\nLow Cost"
    elif climate_class == 'High Affordable\\nHigh Cost':
        display_label = "High Affordable\nHigh Cost"  
    elif climate_class == 'Low Affordable\\nLow Cost':
        display_label = "Low Affordable\nLow Cost"
    else:  # 'Low Affordable\\nHigh Cost'
        display_label = "Low Affordable\nHigh Cost"
    
    node_labels.append(display_label)
    node_colors.append(category_colors[climate_class])

print(f"创建了 {len(node_labels)} 个节点，仿照output.png的标签样式")

# 创建节点索引映射字典
node_indices = {label: i for i, label in enumerate(node_labels)}

# 为空标签节点创建特殊映射
region_start_idx = 0
ideal_start_idx = len(all_regions)
baseline_start_idx = len(all_regions) + len(all_ideal_classes)
climate_start_idx = len(all_regions) + len(all_ideal_classes) + len(all_baseline_classes)

def get_node_index(layer, item):
    """获取节点索引"""
    if layer == 'region':
        return all_regions.index(item)
    elif layer == 'ideal':
        return ideal_start_idx + all_ideal_classes.index(item)
    elif layer == 'baseline':
        return baseline_start_idx + all_baseline_classes.index(item)
    else:  # climate
        return climate_start_idx + all_climate_classes.index(item)

# =============================================================================
# 4. 构建链接数据 - 保持region颜色的连续性
# =============================================================================

# 初始化链接列表
source_indices = []
target_indices = []
link_values = []
link_colors = []

print("开始构建链接关系...")

# --- 第1层到第2层的链接：Region -> Ideal ---
region_to_ideal = sankey_df.groupby(['region', 'ideal_class']).size().reset_index(name='count')
for _, row in region_to_ideal.iterrows():
    source_idx = get_node_index('region', row['region'])
    target_idx = get_node_index('ideal', row['ideal_class'])
    
    source_indices.append(source_idx)
    target_indices.append(target_idx)
    link_values.append(row['count'])
    # 使用稍微透明的区域颜色，仿照output.png的流动效果
    link_colors.append(region_colors[row['region']].replace('0.75)', '0.6)'))

print(f"第1层到第2层链接: {len(region_to_ideal)} 条")

# --- 第2层到第3层的链接：Ideal -> Baseline ---
ideal_to_baseline = sankey_df.groupby(['region', 'ideal_class', 'baseline_class']).size().reset_index(name='count')
for _, row in ideal_to_baseline.iterrows():
    source_idx = get_node_index('ideal', row['ideal_class'])
    target_idx = get_node_index('baseline', row['baseline_class'])
    
    source_indices.append(source_idx)
    target_indices.append(target_idx)
    link_values.append(row['count'])
    # 使用区域颜色保持连续性
    link_colors.append(region_colors[row['region']].replace('0.75)', '0.6)'))

print(f"第2层到第3层链接: {len(ideal_to_baseline)} 条")

# --- 第3层到第4层的链接：Baseline -> Climate Stress ---
baseline_to_climate = sankey_df.groupby(['region', 'baseline_class', 'climate_class']).size().reset_index(name='count')
for _, row in baseline_to_climate.iterrows():
    source_idx = get_node_index('baseline', row['baseline_class'])
    target_idx = get_node_index('climate', row['climate_class'])
    
    source_indices.append(source_idx)
    target_indices.append(target_idx)
    link_values.append(row['count'])
    # 使用区域颜色保持连续性
    link_colors.append(region_colors[row['region']].replace('0.75)', '0.6)'))

print(f"第3层到第4层链接: {len(baseline_to_climate)} 条")
print(f"总链接数: {len(source_indices)}")

# =============================================================================
# 5. 创建Nature风格的桑基图 - 仿照output.png
# =============================================================================

# 创建桑基图对象
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,  # 较小的节点间距，仿照output.png的紧凑感
        thickness=25,  # 较细的节点厚度，更加优雅
        line=dict(color="rgba(200, 200, 200, 0.5)", width=0.8),  # 很淡的边框
        label=node_labels,
        color=node_colors,
        # 仿照output.png的四列布局，稍微调整间距
        x=[0.05 if i < len(all_regions) else
           0.35 if i < len(all_regions) + len(all_ideal_classes) else
           0.65 if i < len(all_regions) + len(all_ideal_classes) + len(all_baseline_classes) else
           0.95
           for i in range(len(node_labels))],
        y=[i/(max(len(all_regions)-1, 1)) if i < len(all_regions) else
           (i-len(all_regions))/(max(len(all_ideal_classes)-1, 1)) if i < len(all_regions) + len(all_ideal_classes) else
           (i-len(all_regions)-len(all_ideal_classes))/(max(len(all_baseline_classes)-1, 1)) if i < len(all_regions) + len(all_ideal_classes) + len(all_baseline_classes) else
           (i-len(all_regions)-len(all_ideal_classes)-len(all_baseline_classes))/(max(len(all_climate_classes)-1, 1))
           for i in range(len(node_labels))]
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=link_values,
        color=link_colors,
        line=dict(color="rgba(255, 255, 255, 0.3)", width=0.3)  # 很淡的链接边框，仿照output.png
    )
)])

# =============================================================================
# 6. 应用output.png的简洁样式设置
# =============================================================================

fig.update_layout(
    font=dict(
        family="Arial",  # Nature期刊标准字体
        size=18,         # 适中的字体大小，仿照output.png
        color="rgb(60, 60, 60)"  # 深灰色文字，不是纯黑
    ),
    plot_bgcolor='rgb(255, 255, 255)',    # 纯白背景，仿照output.png
    paper_bgcolor='rgb(255, 255, 255)',   # 纯白纸张背景
    width=1200,              # 稍微窄一些，仿照output.png的比例
    height=800,              # 适中的高度
    margin=dict(
        l=50,   # 较小的边距，更加紧凑
        r=150,  # 右边距大一些，为标签留空间
        t=30,   # 较小的上边距
        b=30    # 较小的下边距
    ),
    showlegend=False  # 不显示图例，符合output.png的简洁风格
)

# 显示桑基图
fig.show()

# =============================================================================
# 7. 打印样式优化摘要
# =============================================================================

print(f"\n=== Nature风格桑基图（仿照output.png）样式摘要 ===")
print(f"✓ 采用四色主配色方案：青色、橙色、粉红色、金黄色")
print(f"✓ 简洁的节点设计：厚度25，间距20")
print(f"✓ 优雅的流动效果：链接透明度0.6，边框极淡")
print(f"✓ 紧凑的布局：1200×800尺寸，类似output.png比例")
print(f"✓ 清晰的标签系统：左侧区域标签，右侧类别标签")

print(f"\n数据统计:")
print(f"总岛屿数: {len(sankey_df)}")
print(f"涉及区域数: {len(all_regions)}")
print(f"总节点数: {len(node_labels)}")
print(f"总链接数: {len(source_indices)}")

print(f"\n=== 仿照output.png样式的Nature风格桑基图完成 ===")
print("✓ 简洁优雅的颜色方案")
print("✓ 清晰的视觉层次")
print("✓ 期刊级别的图形质量")
print("✓ 保持完整的数据流追踪")