import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import folium_static
import plotly.express as px
import numpy as np

# ----------------------------------------------------------------------
# 1. 設定與數據載入 (Configuration and Data Loading)
# ----------------------------------------------------------------------

# 頁面基本設定
st.set_page_config(
    page_title="PM2.5 預測與視覺化平台",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 模擬數據載入 (Streamlit 建議使用 @st.cache_data 提高性能)
@st.cache_data
def load_data():
    """載入 CSV 和 GeoJSON 檔案，並在找不到時使用模擬數據。"""
    df_raw = pd.DataFrame()
    geojson_data = None
    # 目標縣市，用於篩選和模擬數據
    target_cities = ['臺北', '新北', '桃園', '臺中', '高雄']
    
    # 🚨 更新檔案名稱為用戶上傳的名稱 🚨
    # 如果您未來改回 air_quality_raw.csv，請修改這裡
    file_path = 'air_quality_raw (1).csv' 
    
    # --- 嘗試載入 air_quality_raw (1).csv ---
    try:
        # 載入原始 PM2.5 數據 (用於折線圖)
        df_raw = pd.read_csv(file_path)
        
        # 確保欄位名稱正確轉換 (根據 CSV 預覽: 時間, 測站名稱, PM2.5)
        df_raw.rename(columns={
            '時間': 'Timestamp', 
            '測站名稱': 'City', 
            'PM2.5': 'PM25_VALUE',
            '溫度': 'Temperature',  # 雖然目前沒用，但先轉換
            '濕度': 'Humidity'    # 雖然目前沒用，但先轉換
        }, inplace=True)
        
        # 數據清理與格式化
        df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'])
        
        # 篩選只保留目標縣市的數據
        df_raw = df_raw[df_raw['City'].isin(target_cities)].copy()
        
        if df_raw.empty:
             st.warning(f"⚠️ 找到了 {file_path}，但數據中不包含目標縣市 ({', '.join(target_cities)}) 或資料為空。")
        else:
             st.success(f"✅ 數據檔案 '{file_path}' 載入成功，正在使用真實數據。")
             
    except FileNotFoundError:
        # --------------------------------------------------
        # FALLBACK: 找不到檔案時，自動生成一週的模擬數據
        # --------------------------------------------------
        st.error(f"❌ 錯誤：找不到 '{file_path}'。正在使用**模擬數據**以維持程式運行。")
        
        # 創建模擬時間序列 (7天，每小時一次)
        num_records = 24 * 7 * len(target_cities)
        timestamps = pd.to_datetime(pd.date_range('2025-11-21 00:00', periods=24*7, freq='H')).repeat(len(target_cities))[:num_records]
        
        # 創建模擬城市序列
        cities = np.tile(target_cities, 24 * 7)[:num_records]
        
        # 創建模擬 PM2.5 數據 (加入一些隨機和週期性變化)
        np.random.seed(42)
        random_noise = np.random.uniform(-10, 10, size=num_records)
        base_pm25 = 40 + np.sin(np.linspace(0, 4 * np.pi, num_records)) * 15 + random_noise
        pm25_values = np.clip(base_pm25, 10, 80).astype(int) # 限制在 10 到 80 之間
        
        df_raw = pd.DataFrame({
            'Timestamp': timestamps,
            'City': cities,
            'PM25_VALUE': pm25_values,
            'Temperature': np.random.uniform(15, 30, size=num_records),
            'Humidity': np.random.uniform(50, 90, size=num_records)
        })
        
    except Exception as e:
        st.error(f"❌ 載入 '{file_path}' 時發生錯誤: {e}")
        # 如果載入真實數據失敗，為了確保折線圖頁面能運行，再次執行模擬數據生成
        st.info("嘗試使用模擬數據作為備援。")
        
        # 創建模擬時間序列 (7天，每小時一次)
        num_records = 24 * 7 * len(target_cities)
        timestamps = pd.to_datetime(pd.date_range('2025-11-21 00:00', periods=24*7, freq='H')).repeat(len(target_cities))[:num_records]
        cities = np.tile(target_cities, 24 * 7)[:num_records]
        np.random.seed(42)
        random_noise = np.random.uniform(-10, 10, size=num_records)
        base_pm25 = 40 + np.sin(np.linspace(0, 4 * np.pi, num_records)) * 15 + random_noise
        pm25_values = np.clip(base_pm25, 10, 80).astype(int)
        
        df_raw = pd.DataFrame({
            'Timestamp': timestamps,
            'City': cities,
            'PM25_VALUE': pm25_values,
            'Temperature': np.random.uniform(15, 30, size=num_records),
            'Humidity': np.random.uniform(50, 90, size=num_records)
        })
    
    # --- 載入 GeoJSON (保持不變) ---
    try:
        with open('data/city_data.geojson', 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.warning("GeoJSON 文件 'data/city_data.geojson' 載入失敗，但地圖功能不依賴此檔案。")
    except Exception as e:
        st.error(f"載入 'data/city_data.geojson' 時發生錯誤: {e}")
        
    return df_raw, geojson_data

# 載入所有數據
df_raw, geojson_data = load_data()


# ----------------------------------------------------------------------
# 2. 應用程式結構 (App Structure - 側邊欄與頁面導航)
# ----------------------------------------------------------------------

st.sidebar.title("導航選單")
page = st.sidebar.radio("請選擇功能頁面：", [
    "首頁：專案介紹", 
    "縣市預測地圖", 
    "縣市折線圖", 
    "模型績效排行"
])


# ----------------------------------------------------------------------
# 3. 頁面函式 (Page Functions)
# ----------------------------------------------------------------------

# --------------------
# 3.1 首頁：專案介紹
# --------------------
def page_home():
    st.title("☁️ 永續城市空氣品質與氣候變遷預測平台")
    st.markdown("""
        本平台旨在透過數據科學和機器學習技術，對台灣各縣市的 PM2.5 濃度進行預測與視覺化分析。
        專案響應聯合國永續發展目標 (SDGs)，特別關注以下兩項：
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SDG 11：永續城市與社區")
        st.markdown("""
            - **目標：** 透過 PM2.5 預警，協助城市管理者了解空氣污染熱點。
            - **貢獻：** 提供直觀的地理視覺化，使決策者能更有效地分配環保資源，建立健康的居住環境。
        """)
        try:
            # 檢查圖片載入，如果失敗會顯示警告
            st.image("images/sdg11.png", caption="永續城市與社區", use_column_width=True)
        except Exception:
            st.warning("⚠️ 圖片 images/sdg11.png 載入失敗。請檢查檔案是否存在於 images/ 資料夾。")

    with col2:
        st.subheader("SDG 13：氣候行動")
        st.markdown("""
            - **目標：** 探索氣象因子（溫度、濕度）與 PM2.5 濃度的關聯。
            - **貢獻：** 數據分析結果有助於理解氣候變遷對空氣品質的潛在影響，支持氣候調適策略的制定。
        """)
        try:
            # 檢查圖片載入，如果失敗會顯示警告
            st.image("images/sdg13.png", caption="氣候行動", use_column_width=True)
        except Exception:
            st.warning("⚠️ 圖片 images/sdg13.png 載入失敗。請檢查檔案是否存在於 images/ 資料夾。")

    st.markdown("---")
    st.subheader("系統整合與技術棧")
    st.info("本平台由 Streamlit 構建，前端整合 Folium (地圖)、Plotly (圖表) 和 Pandas (數據處理)。")


# --------------------
# 3.2 縣市預測地圖 (已修改為點狀圖)
# --------------------
def page_map():
    st.title("🗺️ 縣市預測地圖：PM2.5 濃度點位分佈")
    st.info("展示各縣市 PM2.5 濃度點位。點位顏色越深/點越大，代表污染程度越高。")

    # --- 1. 縣市中心點座標查找表 (用於繪製點位) ---
    city_coords = {
        '臺北': [25.033, 121.565], # 臺北市
        '新北': [25.01, 121.46],  # 新北市
        '桃園': [24.99, 121.31],  # 桃園市
        '臺中': [24.14, 120.67],  # 臺中市
        '高雄': [22.62, 120.31]   # 高雄市
    }
    
    # --- 2. 模擬預測數據 ---
    
    target_cities = list(city_coords.keys())
    np.random.seed(42) # 保持結果一致
    
    df_map_data = pd.DataFrame({
        'City': target_cities,
        # 模擬 PM2.5 預測值 (0-80 之間)
        'Predicted_PM25': np.random.randint(15, 80, size=len(target_cities))
    })

    # 合併坐標
    df_map_data['Lat'] = df_map_data['City'].map(lambda x: city_coords.get(x, [None, None])[0])
    df_map_data['Lon'] = df_map_data['City'].map(lambda x: city_coords.get(x, [None, None])[1])
    
    # 移除坐標為 None 的行
    df_map_data.dropna(subset=['Lat', 'Lon'], inplace=True)

    # --- 3. 地圖繪製核心邏輯 (使用 CircleMarker) ---

    # 設置地圖中心點 (台灣西海岸中部，調整 zoom_start 以放大視角，確保所有點都能看到)
    # zoom_start=8 是一個較好的視角
    m = folium.Map(location=[23.5, 120.9], zoom_start=8, tiles="CartoDB positron")

    # 定義顏色映射函數 (PM2.5 越高，顏色越紅)
    def get_color(pm25):
        if pm25 >= 60:
            return '#E31A1C' # 紅色 (高污染)
        elif pm25 >= 45:
            return '#FF7F00' # 橘色 (中高污染)
        elif pm25 >= 30:
            return '#FFD700' # 黃色 (中等)
        else:
            return '#1F78B4' # 藍色 (良好)

    # 迭代數據，添加圓形標記
    for index, row in df_map_data.iterrows():
        pm25 = row['Predicted_PM25']
        color = get_color(pm25)
        
        # 使用 CircleMarker 繪製點位，大小與 PM2.5 相關
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=np.log(pm25) * 4, # 點的大小基於 PM2.5 濃度對數 (讓變化不要太劇烈)
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{row['City']}</b><br>PM2.5 預測值: {pm25:.2f} µg/m³"
        ).add_to(m)

    # 顯示地圖
    folium_static(m, width=900, height=600)
    
    # 顯示顏色圖例
    st.caption("點位圖例：點位大小與 PM2.5 濃度成正比。顏色越暖，濃度越高。")


# --------------------
# 3.3 縣市折線圖 (趨勢分析)
# --------------------
def page_line_chart():
    st.title("📊 縣市 PM2.5 歷史趨勢分析")
    st.info("選擇一個縣市，觀察其 PM2.5 歷史變化與單一測站的最新預測點。")

    # 檢查數據是否載入成功
    if df_raw.empty:
        st.error("數據缺失，無法繪製圖表。請檢查數據載入部分。")
        return

    # 側邊欄選擇器
    all_cities = df_raw['City'].unique()
    # 確保城市列表非空
    if not list(all_cities):
        st.warning("數據中找不到縣市 (City) 名稱，請檢查載入數據的 '測站名稱' 欄位。")
        return
        
    selected_city = st.selectbox("選擇縣市:", all_cities)

    # 確保選定的城市數據非空
    df_city = df_raw[df_raw['City'] == selected_city].copy()
    if df_city.empty:
        st.warning(f"找不到 {selected_city} 的數據。")
        return
    
    # --- 模擬下一小時的預測值 ---
    
    # 找出最新的時間戳
    latest_time = df_city['Timestamp'].max()
    next_time = latest_time + pd.Timedelta(hours=1)
    
    # 根據最新值模擬一個下一小時的預測值 (±5)
    try:
        # 使用 iloc[0] 取得單一值
        latest_pm25 = df_city[df_city['Timestamp'] == latest_time]['PM25_VALUE'].iloc[0]
        # 預測值範圍在 [0, 100]
        predicted_pm25 = max(0, min(100, latest_pm25 + np.random.uniform(-5, 5)))
    except IndexError:
        st.error("所選城市數據異常，無法計算最新值。")
        return
        
    # 創建預測點 DataFrame
    df_prediction = pd.DataFrame({
        'Timestamp': [next_time],
        'PM25_VALUE': [predicted_pm25]
    })
    
    # 繪製 Plotly 折線圖
    fig = px.line(df_city, 
                  x='Timestamp', 
                  y='PM25_VALUE', 
                  title=f'{selected_city} PM2.5 歷史濃度趨勢',
                  labels={'PM25_VALUE': 'PM2.5 濃度 (μg/m³)', 'Timestamp': '時間'},
                  color_discrete_sequence=['#3498db']) 

    # 加上預測點
    fig.add_scatter(x=df_prediction['Timestamp'], 
                    y=df_prediction['PM25_VALUE'], 
                    mode='markers', 
                    marker=dict(size=15, color='red', symbol='circle'),
                    name='下一小時預測值')
    
    # 調整佈局
    fig.update_layout(xaxis_title="時間", 
                      yaxis_title="PM2.5 濃度 (μg/m³)",
                      hovermode="x unified")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 顯示預測結果
    st.markdown(f"**下一小時 ({next_time.strftime('%Y-%m-%d %H:%M')}) 預測值：** <span style='color:red; font-size: 1.2em;'>{predicted_pm25:.2f} μg/m³</span>", unsafe_allow_html=True)


# --------------------
# 3.4 模型績效排行
# --------------------
def page_model_performance():
    st.title("🏆 模型績效排行與比較")
    st.info("比較不同機器學習模型在 PM2.5 預測任務上的表現。")

    # 模擬模型績效數據 (RMSE: Root Mean Squared Error)
    df_models = pd.DataFrame({
        'Model': ['Baseline (簡單平均)', '線性迴歸 (Linear Regression)', 'XGBoost', 'LightGBM', 'Ensemble Model'],
        'RMSE': [25.5, 12.8, 9.2, 8.5, 8.3],
        'R2 Score': [0.0, 0.75, 0.85, 0.88, 0.89]
    }).sort_values(by='RMSE', ascending=True).reset_index(drop=True)
    
    df_models.index = df_models.index + 1
    
    st.subheader("模型 RMSE 績效比較表")
    # 使用 format 參數讓數值顯示更美觀
    st.dataframe(df_models.style.format({
        'RMSE': "{:.2f}", 
        'R2 Score': "{:.2f}"
    }).highlight_min(subset=['RMSE'], color='lightgreen').highlight_max(subset=['R2 Score'], color='lightgreen'), 
                 use_container_width=True)

    st.markdown("---")
    
    # 繪製 Plotly 長條圖
    fig = px.bar(df_models, 
                 x='Model', 
                 y='RMSE', 
                 title='模型 RMSE 誤差值長條圖',
                 text_auto='.2f', # 自動顯示數值，保留兩位小數
                 color='RMSE',
                 color_continuous_scale=px.colors.sequential.Plasma_r) # 顏色越低越好

    fig.update_layout(xaxis_title="機器學習模型", 
                      yaxis_title="PM2.5 預測 RMSE",
                      uniformtext_minsize=8, 
                      uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("RMSE 越低，模型的預測誤差越小，性能越好。")


# ----------------------------------------------------------------------
# 4. 主程式運行 (Main Execution)
# ----------------------------------------------------------------------

if page == "首頁：專案介紹":
    page_home()
elif page == "縣市預測地圖":
    page_map()
elif page == "縣市折線圖":
    page_line_chart()
elif page == "模型績效排行":
    page_model_performance()
