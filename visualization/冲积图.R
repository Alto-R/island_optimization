# 步骤 1: 安装并加载必要的 R 包
# 如果您还没有安装这些包，请取消下面这些行的注释并运行
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("ggalluvial")
# install.packages("tidyr")
# install.packages("RColorBrewer")

library(dplyr)
library(ggplot2)
library(ggalluvial)
library(tidyr)
library(RColorBrewer)

# 步骤 2: 读取您的数据
# 确保 CSV 文件位于您的 R 工作目录中，或提供完整路径
df <- read.csv("chongji_for_R_2.csv", stringsAsFactors = FALSE)

# 步骤 3: 数据预处理和清理
# 查看数据结构
print("数据基本信息:")
print(head(df))
print(str(df))

# 清理分类标签中的换行符，替换为空格
df$ideal_class <- gsub("\n", " ", df$ideal_class)
df$baseline_class <- gsub("\n", " ", df$baseline_class)
df$climate_class <- gsub("\n", " ", df$climate_class)

# 为四列冲积图准备宽格式数据（Region -> Ideal -> Baseline -> Climate）
df_wide <- df %>%
  select(island_id, region, ideal_class, baseline_class, climate_class) %>%
  rename(
    Region = region,
    Ideal = ideal_class,
    Baseline = baseline_class,
    Climate = climate_class
  )

# 查看数据汇总
print("数据汇总:")
print(head(df_wide))
print(paste("数据维度:", paste(dim(df_wide), collapse = " x ")))

# 步骤 4: 动态创建颜色映射
# 自动获取数据中的所有唯一区域
unique_regions <- sort(unique(df$region))
print(paste("数据中包含的区域:", paste(unique_regions, collapse = ", ")))

# 自动获取所有唯一分类
unique_classes <- unique(c(df$ideal_class, df$baseline_class, df$climate_class))
print(paste("数据中包含的分类:", paste(unique_classes, collapse = ", ")))

# 为分类定义颜色
class_colors <- c(
  "High Affordable High Cost" = "#FF6B6B",  # 红色 - 高可负担高成本
  "Low Affordable High Cost" = "#FF8E8E",   # 浅红色 - 低可负担高成本
  "Low Affordable Low Cost" = "#854ecdff",    # 青色 - 低可负担低成本
  "High Affordable Low Cost" = "#d14545ff"    # 蓝色 - 高可负担低成本
)

# 创建综合颜色映射（包含区域和分类的所有类别）
# 获取所有需要着色的类别（第1列是区域，第2-4列是分类）
all_categories <- unique(c(unique_regions, unique_classes))
print(paste("所有需要着色的类别数量:", length(all_categories)))

# 为所有类别分配不同颜色
n_categories <- length(all_categories)
if (n_categories <= 12) {
  # 使用Set3调色板
  color_palette <- brewer.pal(min(max(n_categories, 3), 12), "Set3")
} else {
  # 使用更多颜色
  color_palette <- rainbow(n_categories, s = 0.7, v = 0.8)  # 调整饱和度和亮度
}

# 创建颜色映射
all_colors <- setNames(color_palette[1:n_categories], all_categories)

print("颜色分配:")
for(i in 1:length(all_colors)) {
  print(paste(names(all_colors)[i], ":", all_colors[i]))
}

# 步骤 5: 为实现每列方框不同颜色，需要先转换数据格式
# 创建长格式数据，用于ggalluvial的另一种语法
df_long_alluvial <- df %>%
  select(island_id, region, ideal_class, baseline_class, climate_class) %>%
  gather(key = "variable", value = "value", -island_id) %>%
  mutate(
    variable = factor(variable,
                     levels = c("region", "ideal_class", "baseline_class", "climate_class"),
                     labels = c("Region", "Ideal", "Baseline", "Climate"))
  )

print("长格式数据示例:")
print(head(df_long_alluvial))

# 步骤 6: 创建四列冲积图（Region -> Ideal -> Baseline -> Climate）
# 每列的不同类别用不同颜色，参照参考图样式，修复空隙问题
alluvial_four_column <- ggplot(df_long_alluvial,
                              aes(x = variable, stratum = value, alluvium = island_id, fill = value)) +
  geom_alluvium(alpha = 0.7,          # 流向线透明度，稍微提高以填补空隙
                decreasing = FALSE,   # 不按大小重新排序
                width = 1/6,         # 增加流向线宽度，减少空隙
                knot.pos = 0.5,      # 控制流向线的弯曲点位置
                curve_type = "xspline") + # 使用更平滑的曲线类型
  geom_stratum(width = 1/6,          # 稍微增加方框宽度以减少间隙
               alpha = 0.9,          # 提高方框透明度
               decreasing = FALSE,   # 不重新排序
               color = "white",      # 方框边框色为白色
               size = 0.2) +        # 减小边框粗细，减少视觉间隙
  geom_text(stat = "stratum",
            aes(label = after_stat(stratum)),
            size = 2.6,            # 稍微减小文字大小以适应更窄的方框
            color = "black",       # 文字颜色为黑色
            fontface = "bold",     # 文字粗体
            family = "Arial") +    # 使用Arial字体
  scale_x_discrete(expand = c(0.05, 0.05)) +  # 减小左右边距，让图形更紧凑
  scale_fill_manual(values = all_colors,      # 使用为所有类别分配的颜色
                    name = "Category") +       # 图例标题
  labs(x = "",
       y = "") +                 # 移除标题和轴标签，符合nature风格
  theme_minimal() +
  theme(
    text = element_text(family = "Arial", size = 12),  # 全局Arial字体
    axis.text.x = element_text(size = 14, face = "bold", family = "Arial"),  # x轴标签
    axis.text.y = element_blank(),     # 隐藏y轴标签
    axis.ticks = element_blank(),      # 隐藏坐标轴刻度
    panel.grid = element_blank(),      # 隐藏网格
    panel.background = element_blank(), # 移除背景
    plot.background = element_blank(),  # 移除绘图背景
    legend.position = "none"           # 隐藏图例，参照参考图
  )

print("生成四列冲积图...")
print(alluvial_four_column)

print("保存图表...")

# 保存四列冲积图
ggsave("alluvial_four_column.png", alluvial_four_column,
       width = 8, height = 8, dpi = 300, bg = "white")

# 也可以保存为PDF格式以便后续编辑
# ggsave("alluvial_four_column.pdf", alluvial_four_column,
#        width = 16, height = 10, dpi = 300, bg = "white")

print("图表已保存完成！")
print("生成的文件:")
print("- alluvial_four_column.png: 四列冲积图（PNG格式）")
# print("- alluvial_four_column.pdf: 四列冲积图（PDF格式）")
print("图表展示了岛屿从地理区域到不同情景分类的流向变化")

