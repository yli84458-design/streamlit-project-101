# 程式碼說明：這個腳本用來執行 PM2.5 數據的探索性分析 (EDA)，並生成三種報告圖表。
# 它假設您已經上傳了包含 EPA 和 LASS 合併數據的 'all_pm25_7days.csv' 檔案。

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
import os

# ----------------------------------------------------------------------
# [終極方案] 1. 中文字體設定 (使用 apt-get 系統安裝)
# ----------------------------------------------------------------------

# 檢查是否在 Colab 環境 (為了穩定性，請在 Colab 環境中執行此腳本)
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    print("--- 正在使用 apt-get 安裝系統級中文字體 (WenQuanYi Zen Hei)... ---")
    os.system('apt-get -y install fonts-wqy-zenhei')
    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        # 設定字體名稱
        plt.rcParams['font.sans-serif'] = 'WenQuanYi Zen Hei' 
        plt.rcParams['axes.unicode_minus'] = False # 解決負號亂碼問題
        print("--- 字體安裝成功，已設定 Matplotlib 使用 'WenQuanYi Zen Hei'。 ---")
    else:
        print("--- 警告：中文字體安裝路徑異常，圖表可能無法顯示中文。 ---")
else:
    # 嘗試使用系統預設字體
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    print("--- 非 Colab 環境，嘗試使用系統預設中文字體。 ---")


# ======================================================================
# 2. 讀取數據
# ======================================================================

DATA_FILE = 'all_pm25_7days.csv'
print(f"\n📥 2. 嘗試讀取合併後的資料檔案: {DATA_FILE}...")

try:
    # 讀取數據
    df = pd.read_csv(DATA_FILE)
    df['Timestamp_Aligned_Hour'] = pd.to_datetime(df['Timestamp_Aligned_Hour'])
    
    # 確保關鍵欄位是數值型
    for col in ['LASS_PM25', 'LASS_Temp', 'LASS_Humid', 'EPA_PM25']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['LASS_PM25', 'LASS_Temp', 'LASS_Humid', 'EPA_PM25'], inplace=True)
    df.set_index('Timestamp_Aligned_Hour', inplace=True)

    print(f"✓ 數據讀取成功！有效筆數: {len(df):,}")
    print(df.head())

except FileNotFoundError:
    print(f"❌ 嚴重錯誤：找不到 {DATA_FILE} 檔案。請確認您已執行資料合併腳本！")
    exit()
except Exception as e:
    print(f"❌ 讀取或清理數據時發生錯誤: {e}")
    exit()

# ----------------------------------------------------------------------
# 3. 任務一：PM2.5 時間序列分析 (Time Series Plot)
# ----------------------------------------------------------------------

print("\n--- 3. 繪製 PM2.5 時間序列圖 (優化) ---")

plt.figure(figsize=(15, 6))

# 繪製 LASS PM2.5 (細線，強調連續性)
plt.plot(df.index, df['LASS_PM25'], 
         label='LASS PM2.5 (感測器平均)', 
         color='#FF6347', 
         alpha=0.7, 
         linewidth=1.5,
         marker='.', markersize=4)

# 繪製 EPA PM2.5 (粗線，強調官方數據點)
plt.plot(df.index, df['EPA_PM25'], 
         label='EPA PM2.5 (官方測站平均)', 
         color='#1E90FF', 
         alpha=0.8, 
         linewidth=2.5,
         marker='o', markersize=6)


plt.title('過去七天 PM2.5 時間序列趨勢：LASS 與 EPA 數據對比', fontsize=18, fontweight='bold')
plt.xlabel('時間 (小時)', fontsize=14)
plt.ylabel('PM2.5 濃度 ($\mu g/m^3$)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pm25_time_series.png')
print("✅ PM2.5 時間序列圖繪製完成 (pm25_time_series.png)。")


# ----------------------------------------------------------------------
# 4. 任務二：氣象特徵 vs PM2.5 散布圖 (Scatter Plot)
# ----------------------------------------------------------------------

print("\n--- 4. 繪製氣象特徵 vs PM2.5 散布圖 (優化) ---")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- 圖 1: 溫度 vs PM2.5 ---
sns.regplot(x='LASS_Temp', y='LASS_PM25', data=df, ax=axes[0], 
            scatter_kws={'alpha': 0.4, 's': 20, 'color': '#20B2AA'}, 
            line_kws={'color': '#FF4500'})
axes[0].set_title('溫度 vs PM2.5 散布圖', fontsize=16, fontweight='bold')
axes[0].set_xlabel('溫度 ($^\circ C$)', fontsize=14)
axes[0].set_ylabel('PM2.5 濃度 ($\mu g/m^3$)', fontsize=14)
axes[0].grid(axis='y', linestyle=':', alpha=0.6)

# --- 圖 2: 濕度 vs PM2.5 ---
sns.regplot(x='LASS_Humid', y='LASS_PM25', data=df, ax=axes[1], 
            scatter_kws={'alpha': 0.4, 's': 20, 'color': '#4682B4'},
            line_kws={'color': '#DAA520'})
axes[1].set_title('濕度 vs PM2.5 散布圖', fontsize=16, fontweight='bold')
axes[1].set_xlabel('濕度 (%)', fontsize=14)
axes[1].set_ylabel('PM2.5 濃度 ($\mu g/m^3$)', fontsize=14)
axes[1].grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout(pad=3.0) # 自動調整子圖間距
plt.savefig('meteorological_scatter.png')
print("✅ 氣象特徵 vs PM2.5 散布圖繪製完成 (meteorological_scatter.png)。")


# ----------------------------------------------------------------------
# 5. 任務三：相關係數熱圖 (Correlation Heatmap) - 報告優化
# ----------------------------------------------------------------------

print("\n--- 5. 繪製相關係數熱圖 (優化版) ---")

# 選擇用於計算相關係數的數值欄位
# 注意：我們假設 Wind_Speed, AQI 等欄位可能不存在於您的合併數據中，這裡只使用 LASS/EPA 的核心數據
numeric_cols = ['LASS_PM25', 'LASS_Temp', 'LASS_Humid', 'EPA_PM25']

# 確保欄位存在且是數值型
df_corr = df[numeric_cols].copy()

# 計算相關係數矩陣 (Correlation Matrix)
corr_matrix = df_corr.corr()

plt.figure(figsize=(9, 8))
sns.heatmap(
    corr_matrix, 
    annot=True,          # 顯示數值
    cmap='coolwarm',     # 顏色圖
    fmt=".2f",           # 數值格式
    linewidths=0.5,      # 線寬
    linecolor='black',
    cbar_kws={'label': '相關係數 (Correlation Coefficient)'}
)

plt.title('PM2.5 及其相關特徵之相關係數熱圖', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("✅ 相關係數熱圖繪製完成 (correlation_heatmap.png)。")

print("\n🎉 EDA 腳本執行完畢。共輸出三張圖片 (png 檔案) 到您的工作目錄。")
print("您現在可以檢視這些圖片以獲得深入的洞察。")

# 為了在某些環境中能自動顯示 Matplotlib 圖形，保留 plt.show()
# 但如果您在 Colab/Jupyter 環境中執行，圖片會自動顯示
# plt.show()
