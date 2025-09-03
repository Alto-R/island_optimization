import pandas as pd
import numpy as np
import xarray as xr
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import os
import argparse
import gc

### 获取参数和数据 ###
# 获取岛屿经纬度参数
parser = argparse.ArgumentParser(description="Process island coordinates and population.")
parser.add_argument("--island_lat", type=float, required=True, help="Latitude of the island")
parser.add_argument("--island_lon", type=float, required=True, help="Longitude of the island")
parser.add_argument("--pop", type=int, required=True, help="Population of the island")
args = parser.parse_args()
island_lat = args.island_lat
island_lon = args.island_lon
pop = args.pop
if pop < 500:
    pop = pop
else:
    pop = 500
island_coords = (island_lat, island_lon)

# 获取波浪能数据
def get_wave(data,lat,lon):
    if lon < 0 :
        lon = lon + 360
    else :
        lon = lon
    location_wave = data['WPD'].sel(lat=lat, lon=lon, method='nearest')
    
    wave_power_df = location_wave.to_dataframe().reset_index()
    wave_power_df = wave_power_df[wave_power_df['npt']==1]
    
    last_time = pd.Timestamp("2021-01-01 00:00:00")
    last_row = wave_power_df.iloc[-1].copy()
    last_row['time'] = last_time
    new_row = pd.DataFrame([last_row])
    wave_df = pd.concat([wave_power_df, new_row])
    
    wave_df['time'] = pd.to_datetime(wave_df['time'])
    wave_df.set_index("time", inplace=True) 
    wave_df = wave_df.resample('3h').interpolate(method='linear')
    wave_df['t'] = range(0, len(wave_df))
    
    wave_df.fillna(0, inplace=True)
    wave_df[wave_df < 0] = 0
    
    return wave_df

wave = xr.open_dataset('wave/wave_2020.nc')
wave_df = get_wave(wave,island_lat,island_lon)
wave_power = wave_df['WPD'].values


# 找出最近的终端
LNG_terminals = pd.read_excel('LNG/LNG_Terminals.xlsx')
LNG_terminals = LNG_terminals[LNG_terminals['Status'].isin(['Construction', 'Operating'])]
LNG_terminals = LNG_terminals.dropna(subset=['Latitude', 'Longitude'])
LNG_terminals['Distance_km'] = LNG_terminals.apply(
    lambda row: geodesic(island_coords, (row['Latitude'], row['Longitude'])).kilometers,
    axis=1
)
closest_terminal = LNG_terminals.loc[LNG_terminals['Distance_km'].idxmin()]
LNG_terminal_lat = closest_terminal['Latitude']
LNG_terminal_lon = closest_terminal['Longitude']
LNG_distance = closest_terminal['Distance_km']

# 获取风速数据
def get_wind(data1,data2,lat,lon):
    if lon < 0 :
        lon = lon + 360
    else :
        lon = lon
    location_wind1 = data1['uas'].sel(lat=lat, lon=lon, method='nearest')
    location_wind2 = data2['vas'].sel(lat=lat, lon=lon, method='nearest')
    location_wind = np.sqrt(location_wind1**2 + location_wind2**2)
    
    wind_df = location_wind.to_dataframe(name='windspeed')
    
    last_time = pd.Timestamp("2021-01-01 00:00:00")
    last_wind_speed = wind_df.iloc[-1]['windspeed']
    new_row = pd.DataFrame({'windspeed': last_wind_speed}, index=[last_time])
    wind_df = pd.concat([wind_df, new_row])
    
    wind_df = wind_df.resample('3h').interpolate(method='linear')
    
    wind_df['t'] = range(0, len(wind_df))
    
    wind_df.fillna(0, inplace=True)
    wind_df[wind_df < 0] = 0
    
    return wind_df

folder_path = "CMIP6/MRI_2020_uas"
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc')]
file_paths.sort()
combined_data_2020_1 = xr.open_mfdataset(file_paths, combine='by_coords')

folder_path = "CMIP6/MRI_2020_vas"
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc')]
file_paths.sort()
combined_data_2020_2 = xr.open_mfdataset(file_paths, combine='by_coords')

wind1 = combined_data_2020_1
wind2 = combined_data_2020_2
wind_df = get_wind(wind1,wind2,island_lat,island_lon)

# 释放内存
combined_data_2020_1.close()
combined_data_2020_2.close()
del combined_data_2020_1, combined_data_2020_2
gc.collect()

# 获取波高数据
def get_wave_h(data,lat,lon):
    if lon < 0 :
        lon = lon + 360
    else :
        lon = lon
    location_wave = data['Hs'].sel(lat=lat, lon=lon, method='nearest')
    wave_h_df = location_wave.to_dataframe().reset_index()
    wave_h_df = wave_h_df[wave_h_df['npt']==1]
    last_time = pd.Timestamp("2021-01-01 00:00:00")
    last_row = wave_h_df.iloc[-1].copy()
    last_row['time'] = last_time
    new_row = pd.DataFrame([last_row])
    wave_df = pd.concat([wave_h_df, new_row])
    wave_df['time'] = pd.to_datetime(wave_df['time'])
    wave_df.set_index("time", inplace=True) 
    wave_df = wave_df.resample('3h').interpolate(method='linear')
    wave_df['t'] = range(0, len(wave_df))
    wave_df.fillna(0, inplace=True)
    wave_df[wave_df < 0] = 0
    return wave_df

wave_h = xr.open_dataset('wave/waveheight_2020.nc')
wave_h_df = get_wave_h(wave_h,island_lat,island_lon)

# 修复时间 (转换为3小时时间步)
repair_time = {
    'WT': 336//3, 'PV': 96//3, 'WEC': 336//3  # 原始小时数除以3
}

### 可靠性建模 ###
def failure_probability_wt(ve):
    if ve < 30:
        return 0
    elif ve >= 60:
        return 1.0
    else:
        return (ve - 30) / (60 - 30)

def failure_probability_pv(ve):
    if ve < 40:
        return 0
    elif ve >= 80:
        return 1.0
    else:
        return (ve - 40) / (80 - 40)

def failure_probability_wave(h):
    if h < 5:
        return 0
    elif h >= 20:
        return 1.0
    else:
        return (h - 5) / (20 - 5)

def device_state_simulation(repair_times, wind_df, wave_h_df):
    time_horizon = 2928  # 2020年的3小时时段数 (8784/3)
    # 初始化设备状态数据框，跟踪每个时间步的设备状态
    device_generate = ['WT', 'PV', 'WEC', 'LNG']
    device_states_df = pd.DataFrame(index=range(time_horizon), columns=device_generate)
    
    device_states = {device: 1 for device in device_generate}  # 1 设备正常工作
    time_in_states = {device: 0 for device in device_generate}
    
    rng = np.random.default_rng()
    
    # 风速影响组件是否损坏以及修复过程
    for t in range(time_horizon):
        V = wind_df['windspeed'].iloc[t] 
        h = wave_h_df['Hs'].iloc[t] 
        
        for device in device_generate:
            if device == 'LNG':
                # LNG设备总是保持工作状态，除非被后续的高风速规则影响
                device_states[device] = 1
            else:
                # 只检查当前处于正常状态的设备是否会失效
                if device_states[device] == 1:
                    if device == 'WT':
                        failure = failure_probability_wt(V)
                        if rng.random() < failure:
                            device_states[device] = 0
                            time_in_states[device] = 0  # 重置修复时间计数器
                    elif device == 'PV':
                        failure = failure_probability_pv(V)
                        if rng.random() < failure:
                            device_states[device] = 0
                            time_in_states[device] = 0  
                    elif device == 'WEC':
                        failure = failure_probability_wave(h)
                        if rng.random() < failure:
                            device_states[device] = 0
                            time_in_states[device] = 0 
                
                # 如果设备处于故障状态，计算修复进度
                if device_states[device] == 0:
                    if device == 'WEC':
                        # WEC设施需要波高小于2才能维修
                        if h <= 2 and V <= 20:
                            time_in_states[device] += 1
                    # 只有在风速适宜时才能进行修复工作
                    else:
                        if V <= 20:
                            time_in_states[device] += 1
                    
                    # 检查是否修复完成
                    if time_in_states[device] >= repair_times[device]:
                        device_states[device] = 1  # 修复完成，恢复正常状态
                        time_in_states[device] = 0  # 重置修复时间计数器
            
            # 在当前时间步记录设备状态
            device_states_df.at[t, device] = device_states[device]
    
    # 风速过高导致设备停止运行（不论是否已经失效）
    critical_devices = {'WT', 'PV', 'LNG'}
    for t in range(time_horizon):
        V = wind_df['windspeed'].iloc[t]
        for device in critical_devices:
            if V > 20:
                device_states_df.at[t, device] = 0
    
    return device_states_df


def extract_features_from_scenario(scenario_df):
    """
    为单个场景(DataFrame)提取关键特征。
    """
    features = {}
    devices = scenario_df.columns

    for device in devices:
        downtime = 1 - scenario_df[device] # 1表示停机, 0表示正常
        
        # 特征1: 总停机时间
        features[f'total_downtime_{device}'] = downtime.sum()
        
        # 特征2: 最长连续停机时间
        # 计算连续停机时间块的长度
        consecutive_downtime = downtime.groupby((downtime != downtime.shift()).cumsum()).cumsum()
        features[f'max_consecutive_downtime_{device}'] = consecutive_downtime.max()
        
        # 特征3: 停机次数
        # 当状态从1(正常)变为0(停机)时，算作一次新的停机事件
        features[f'downtime_events_{device}'] = (scenario_df[device].diff() == -1).sum()

    # 特征4: 多设备同时停机
    # 计算至少有两个设备同时停机的时间步数
    features['simultaneous_2_downtime'] = (scenario_df.sum(axis=1) <= (len(devices) - 2)).sum()
    # 计算至少有三个设备同时停机的时间步数
    features['simultaneous_3_downtime'] = (scenario_df.sum(axis=1) <= (len(devices) - 3)).sum()
    
    return features


# 设备失效状态聚类
 # --- 参数设置 ---
num_scenarios_to_generate = 1000 # 生成1000个原始场景用于分析

scenarios_df_list = [device_state_simulation(repair_time, wind_df, wave_h_df) for i in range(num_scenarios_to_generate)]
scenarios_3d_array = np.array([df.values for df in scenarios_df_list])

# --- 步骤 2: 为每个场景提取特征 ---
features_list = [extract_features_from_scenario(df) for df in scenarios_df_list]
features_df = pd.DataFrame(features_list)

# --- 步骤 3a: 处理所有场景都相同的特殊情况 ---
# 检查去重后还剩多少个独特的特征向量
unique_features_df = features_df.drop_duplicates()
if len(unique_features_df) == 1:
    print("聚类分析将被跳过，直接采用这个唯一的场景作为代表。")
    # 直接选择第一个场景作为唯一的代表
    binary_cluster_centers = scenarios_3d_array[0:1] # 取第一个场景，并保持3D形状

else:
    # --- 步骤 3b: 【原流程】确定最佳聚类数量 K ---
    print("\n步骤 3: 发现多个不同场景，正在确定最佳聚类数量 (K)...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # # 方法A: 肘部法则
    # sse = []
    # k_range = range(2, 11)
    # for k in k_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans.fit(features_scaled)
    #     sse.append(kmeans.inertia_)
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(k_range, sse, 'bo-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Sum of Squared Errors (SSE)')
    # plt.title('Elbow Method for Optimal k')
    # plt.grid(True)
    # plt.show()

    # # 方法B: 轮廓系数
    # silhouette_scores = []
    # for k in k_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans.fit(features_scaled)
        
    #     # 安全检查：确保至少有两个簇被找到
    #     if len(np.unique(kmeans.labels_)) < 2:
    #         print(f"警告：对于 k={k}, K-Means只找到了1个簇。无法计算轮廓系数，将跳过。")
    #         silhouette_scores.append(-1) # 用一个无效值标记
    #         continue
        
    #     score = silhouette_score(features_scaled, kmeans.labels_)
    #     silhouette_scores.append(score)

    # plt.figure(figsize=(10, 5))
    # plt.plot(k_range, silhouette_scores, 'ro-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score for Optimal k')
    # plt.grid(True)
    # plt.show()
    
    # try:
    #     n_clusters = int(input("根据图表，请输入您决定的簇数 (k): "))
    # except ValueError:
    #     print("输入无效，使用默认值 k=4")
    #     n_clusters = 4
    
    n_clusters = 4
    # --- 步骤 4: 执行最终聚类并提取代表性场景 ---
    final_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    final_kmeans.fit(features_scaled)
    labels = final_kmeans.labels_
    centroids = final_kmeans.cluster_centers_
    
    representative_indices = []
    for i in range(n_clusters):
        cluster_i_indices = np.where(labels == i)[0]
        if len(cluster_i_indices) == 0: continue
        cluster_features = features_scaled[cluster_i_indices]
        centroid_i = centroids[i]
        distances = np.linalg.norm(cluster_features - centroid_i, axis=1)
        medoid_local_index = np.argmin(distances)
        medoid_global_index = cluster_i_indices[medoid_local_index]
        representative_indices.append(medoid_global_index)

    binary_cluster_centers = scenarios_3d_array[representative_indices]
    

