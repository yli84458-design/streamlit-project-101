# 程式碼說明：這個腳本用來執行 PM2.5 數據的探索性分析 (EDA)，並生成三種報告圖表。
# 我們將使用 pandas 處理數據，並使用 matplotlib 和 seaborn 繪製圖表。

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
import os

# ----------------------------------------------------------------------
# [終極方案] 1. 中文字體設定 (適用於 Colab)
# ----------------------------------------------------------------------

print("--- 正在使用 apt-get 安裝系統級中文字體 (WenQuanYi Zen Hei)... ---")
# 在 Colab 中執行系統指令安裝中文字體
os.system('apt-get -y install fonts-wqy-zenhei')
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

if os.path.exists(font_path):
    # 重新整理字體快取並設定 Matplotlib 參數
    fm.fontManager.addfont(font_path)
    print("--- 字體安裝成功，已加入 Matplotlib ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
    sns.set(font='WenQuanYi Zen Hei')
    sns.set_theme(style="whitegrid", font="WenQuanYi Zen Hei")
else:
    print("警告：字體安裝似乎失敗，將使用預設字體。")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']

# ----------------------------------------------------------------------
# 2. 數據載入與清理 (Data Loading and Cleaning)
# ----------------------------------------------------------------------

print("--- 2. 載入數據 ---")
# *** 確認檔案 air_quality_raw.csv 已上傳到 Colab ***
try:
    df = pd.read_csv('air_quality_raw.csv')
    print(f"數據載入成功！總共有 {len(df)} 筆資料。")
except FileNotFoundError:
    print("錯誤：找不到 'air_quality_raw.csv' 檔案。請先將檔案上傳到 Colab 執行環境中。")
    exit()

# *** 修正欄位名稱 ***
df.rename(columns={
    'PM2.5': 'PM25_VALUE', 
    '溫度': 'Temperature', 
    '濕度': 'Humidity',
    '時間': 'Timestamp',
    '測站名稱': 'StationName'
}, inplace=True)


# 選擇我們需要的欄位進行分析
required_cols = ['PM25_VALUE', 'Temperature', 'Humidity', 'Timestamp']
df_eda = df[required_cols].copy()

# 檢查並處理缺失值
print("\n--- 3. 數據清理與預處理 ---")
df_eda.dropna(subset=['PM25_VALUE', 'Temperature', 'Humidity'], inplace=True)

# 轉換時間戳為日期時間格式並提取小時
df_eda['Timestamp'] = pd.to_datetime(df_eda['Timestamp'])
df_eda['Hour'] = df_eda['Timestamp'].dt.hour

# ----------------------------------------------------------------------
# 3. 任務一：PM2.5 日週期圖 (Daily Cycle Plot)
# ----------------------------------------------------------------------

print("\n--- 4. 繪製 PM2.5 日週期圖 ---")
daily_cycle = df_eda.groupby('Hour')['PM25_VALUE'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Hour', y='PM25_VALUE', data=daily_cycle, marker='o', color='#3498db', linewidth=2)

alert_value = 35
plt.axhline(alert_value, color='red', linestyle='--', alpha=0.7, label=f'PM2.5 警戒線 ({alert_value} μg/m³)')

plt.fill_between(daily_cycle['Hour'], daily_cycle['PM25_VALUE'], alert_value,
                 where=(daily_cycle['PM25_VALUE'] > alert_value),
                 color='red', alpha=0.1, interpolate=True)

plt.title('PM2.5 日週期變化圖：趨勢與警戒值比較', fontsize=16, fontweight='bold')
plt.xlabel('小時 (Hour of Day)', fontsize=12)
plt.ylabel('平均 PM2.5 濃度 (μg/m³)', fontsize=12)
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
print("PM2.5 日週期圖繪製完成。")

# ----------------------------------------------------------------------
# 4. 任務二：氣象特徵 vs PM2.5 散布圖
# ----------------------------------------------------------------------

print("\n--- 5. 繪製氣象特徵 vs PM2.5 散布圖 ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# a) 溫度 (Temperature) vs PM2.5
sns.regplot(x='Temperature', y='PM25_VALUE', data=df_eda, ax=axes[0],
            scatter_kws={'alpha': 0.1, 'color': '#3498db'},
            line_kws={'color': '#e74c3c', 'linewidth': 2})

axes[0].set_title('溫度 vs PM2.5 散布圖：觀察集中趨勢', fontsize=14, fontweight='bold')
axes[0].set_xlabel('溫度 (°C)', fontsize=12)
axes[0].set_ylabel('PM25 濃度 (μg/m³)', fontsize=12)
axes[0].grid(axis='y', linestyle=':', alpha=0.6)


# b) 濕度 (Humidity) vs PM2.5
sns.regplot(x='Humidity', y='PM25_VALUE', data=df_eda, ax=axes[1],
            scatter_kws={'alpha': 0.1, 'color': '#2ecc71'},
            line_kws={'color': '#f39c12', 'linewidth': 2})

axes[1].set_title('濕度 vs PM2.5 散布圖：觀察集中趨勢', fontsize=14, fontweight='bold')
axes[1].set_xlabel('濕度 (%)', fontsize=12)
axes[1].set_ylabel('PM25 濃度 (μg/m³)', fontsize=12)
axes[1].grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout(pad=3.0)
plt.show()
print("氣象特徵 vs PM2.5 散布圖繪製完成。")

# ----------------------------------------------------------------------
# 5. 任務三：相關係數熱圖
# ----------------------------------------------------------------------

print("\n--- 6. 繪製相關係數熱圖 ---")
numeric_cols_for_corr = ['PM25_VALUE', 'Temperature', 'Humidity']
corr_matrix = df_eda[numeric_cols_for_corr].corr()

plt.figure(figsize=(10, 9))
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='viridis',
    fmt=".2f",
    linewidths=.5,
    linecolor='black',
    annot_kws={"fontsize": 10},
)

cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('相關係數 (Correlation Coefficient)', rotation=270, labelpad=15)

plt.title('特徵相關係數熱圖：多維度特徵關係分析', fontsize=16, fontweight='bold')

plt.show()
print("相關係數熱圖繪製完成。")

print("\n所有要求的 EDA 圖表已成功生成。")
