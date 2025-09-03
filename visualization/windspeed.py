import xarray as xr
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# 读取岛屿数据
island = pd.read_csv('filtered_island_1898.csv')

# 读取 NetCDF 数据
def load_nc_files(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc')]
    file_paths.sort()
    return xr.open_mfdataset(file_paths, combine='by_coords')

combined_data_2020_1 = load_nc_files("../CMIP6/MRI_2020_uas")
combined_data_2020_2 = load_nc_files("../CMIP6/MRI_2020_vas")
combined_data_2050_1 = load_nc_files("../CMIP6/MRI_2050_uas")
combined_data_2050_2 = load_nc_files("../CMIP6/MRI_2050_vas")

# 定义处理每个岛屿任务的函数
def process_island(row):
    lat, lon = row['Lat'], row['Long']
    island_id = row['ID']
    
    if lon < 0:
        lon1 = lon + 360
    else:
        lon1 = lon

    # 选取最近的 uas 和 vas 数据
    uas_loc1 = combined_data_2020_1['uas'].sel(lat=lat, lon=lon1, method='nearest')
    vas_loc1 = combined_data_2020_2['vas'].sel(lat=lat, lon=lon1, method='nearest')
    uas_loc2 = combined_data_2050_1['uas'].sel(lat=lat, lon=lon1, method='nearest')
    vas_loc2 = combined_data_2050_2['vas'].sel(lat=lat, lon=lon1, method='nearest')

    # 计算风速
    wind_speed_loc1 = np.sqrt(uas_loc1**2 + vas_loc1**2)
    wind_speed_loc2 = np.sqrt(uas_loc2**2 + vas_loc2**2)

    # 转换为 DataFrame
    df1 = wind_speed_loc1.to_dataframe(name='Wind_Speed_2020')
    df2 = wind_speed_loc2.to_dataframe(name='Wind_Speed_2050')

    # 重置索引以便更好地保存
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    
    # 添加岛屿信息
    df1['Island_ID'] = island_id
    df1['Lat'] = lat
    df1['Long'] = lon
    df2['Island_ID'] = island_id
    df2['Lat'] = lat
    df2['Long'] = lon

    # 创建输出目录（如果不存在）
    os.makedirs('island_windspeed_data', exist_ok=True)
    
    # 保存每个岛屿的风速数据（单独保存2020和2050年）
    df1.to_csv(f'island_windspeed_data/{lat}_{lon}_2020_windspeed.csv', index=False)
    df2.to_csv(f'island_windspeed_data/{lat}_{lon}_2050_windspeed.csv', index=False)

# 使用多进程
if __name__ == "__main__":
    # 将岛屿数据逐行并行处理
    with ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(process_island, [row for _, row in island.iterrows()])