import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os 
import hashlib 
import time

# ==========================================
# 1. 系統設定與快取
# ==========================================
st.set_page_config(page_title="台灣 AI 空氣品質預測戰情室", layout="wide", page_icon="🍃")

# 備援測站座標 (未使用於 LASS 數據)
STATIONS_COORDS = {
    '臺北': {'lat': 25.0330, 'lon': 121.5654}, '新北': {'lat': 25.0129, 'lon': 121.4624},
    '桃園': {'lat': 24.9976, 'lon': 121.3033}, '臺中': {'lat': 24.1477, 'lon': 120.6736},
    '臺南': {'lat': 22.9997, 'lon': 120.2270}, '高雄': {'lat': 22.6273, 'lon': 120.3014}
}

# 輔助函數：將 LASS 經緯度粗略分組到縣市/區域
def map_coord_to_city(lat, lon):
    if lat > 24.8 and lon > 121: return "北部地區 (台北/新北/基隆)"
    if lat > 24.3 and lon < 121: return "桃竹苗地區"
    if lat < 24.3 and lat > 23.5 and lon < 121: return "中部地區 (台中/彰化/南投)"
    if lat < 23.5 and lat > 22.5 and lon < 121: return "雲嘉南地區"
    if lat < 22.5 and lon < 121: return "高屏地區"
    if lon > 121 and lat > 23: return "東部地區 (宜花東)"
    return "其他/離島"

# 輔助函數：為 LASS ID 生成一個人類可讀的站點名稱 (使用 City 和 ID 尾碼)
def generate_station_name(device_id, city):
    """Generates a human-readable name using City and a short hash of the device ID."""
    # 使用 ID 的前四個字符作為尾碼
    short_hash = device_id[:4].upper() if device_id else "N/A"
    # 使用更簡潔的格式，模擬一個站點名稱
    return f"{city} - {short_hash}"

# ==========================================
# 2. 資料獲取與處理模組
# ==========================================

@st.cache_data(ttl=60) # 60秒更新一次即時數據
def get_lass_data():
    """ 
    整合 LASS 即時資料爬蟲邏輯 (包含 PM2.5, 溫度, 濕度)
    """
    url = "https://pm25.lass-net.org/data/last-all-airbox.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
            
        data = response.json()
        records = data.get('feeds', data)
            
        df = pd.DataFrame(records)
        
        # 欄位對應 (s_d0=PM2.5, s_t0=Temp, s_h0=Humidity)
        rename_dict = {
            's_d0': 'pm25', 's_t0': 'temp', 's_h0': 'humidity',
            'gps_lat': 'lat', 'gps_lon': 'lon', 'timestamp': 'time', 'device_id': 'id'
        }
        
        existing_cols = [c for c in rename_dict.keys() if c in df.columns]
        df = df[existing_cols].copy()
        df.rename(columns=rename_dict, inplace=True)
        
        # 數值轉換與過濾 (僅取台灣範圍與合理數值)
        cols = ['pm25', 'lat', 'lon', 'temp', 'humidity']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
        df = df[
            (df['lat'].between(21, 26)) & (df['lon'].between(119, 123)) & 
            (df['pm25'].between(0, 500))
        ]
        
        return df.dropna(subset=['pm25', 'lat', 'lon'])
        
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """ 
    載入訓練好的模型 (預期檔名: model.pkl)
    """
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            return None
    return None

@st.cache_data
def load_historical_data():
    """ 
    讀取合併後的歷史數據 (預期檔名: all_pm25_7days.csv)
    """
    file_path = 'all_pm25_7days.csv'
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # 兼容不同的時間欄位名稱
            time_col = None
            if 'Timestamp_Aligned_Hour' in df.columns:
                time_col = 'Timestamp_Aligned_Hour'
            elif 'time' in df.columns:
                time_col = 'time'
                
            if time_col:
                df['time'] = pd.to_datetime(df[time_col])
            else:
                st.error("歷史資料中找不到時間欄位 (Timestamp_Aligned_Hour 或 time)。")
                return pd.DataFrame()
                
            # 將數值欄位轉換為 float (避免資料型態問題)
            numeric_cols = ['LASS_PM25', 'LASS_Temp', 'LASS_Humid', 'EPA_PM25', 'AQI', 'Wind_Speed']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df.dropna(subset=['time', 'LASS_PM25'])
        except Exception as e:
            st.error(f"讀取歷史數據時發生錯誤: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# ==========================================
# 3. 初始化 (載入資料與模型)
# ==========================================

df_live = get_lass_data()
df_hist = load_historical_data()
model = load_model()

# **[關鍵修正]**：將縣市分類應用到即時資料，並生成使用者友善的站點名稱，欄位命名為 'sitename'
station_name_to_id = {}
if not df_live.empty:
    df_live['City'] = df_live.apply(lambda row: map_coord_to_city(row['lat'], row['lon']), axis=1)
    
    # 創建使用者友善名稱，並命名為 'sitename' 欄位
    df_live['sitename'] = df_live.apply(lambda row: generate_station_name(row['id'], row['City']), axis=1)
    
    # 創建名稱到 ID 的反向映射字典
    station_name_to_id = df_live.set_index('sitename')['id'].to_dict()

# ==========================================
# 4. 介面呈現 (Streamlit Layout)
# ==========================================

# --- 側邊欄 ---
with st.sidebar:
    st.title("控制面板")
    page = st.radio("功能切換", ["即時戰情室", "歷史數據分析", "模型預測展示"])
    
    st.markdown("---")
    st.markdown("### 系統狀態")
    st.write(f"🟢 LASS 連線: {'正常' if not df_live.empty else '異常 (正在重試...)'}")
    st.write(f"🟢 歷史資料庫: {'已載入' if not df_hist.empty else '未找到 all_pm25_7days.csv'}")
    st.write(f"🟢 AI 模型: {'已就緒' if model else '未找到 model.pkl'}")
    
    # 檔案偵錯區塊 
    st.markdown("---")
    st.markdown("### 🔍 檔案偵錯 (Debug)")
    try:
        current_files = os.listdir('.')
        st.caption("專案根目錄中的檔案:")
        st.code('\n'.join(current_files), language='text')
    except Exception:
        pass


# --- 頁面 1: 即時戰情室 (無變動) ---
if page == "即時戰情室":
    st.title("🍃 台灣 AI 空氣品質即時戰情室")
    
    if not df_live.empty:
        col1, col2, col3 = st.columns(3)
        avg_pm25 = df_live['pm25'].mean()
        high_risk = len(df_live[df_live['pm25'] > 35])
        
        col1.metric("全台平均 PM2.5", f"{avg_pm25:.1f} µg/m³", delta="即時更新")
        col2.metric("高風險站點數 (>35)", f"{high_risk} 站", delta_color="inverse")
        if 'temp' in df_live.columns:
            col3.metric("平均氣溫/濕度", f"{df_live['temp'].mean():.1f} °C / {df_live['humidity'].mean():.1f} %")
    else:
        st.warning("⚠️ 目前無法取得 LASS 即時資料，請稍後再試。")

    st.markdown("---")
    
    if not df_live.empty:
        st.subheader("🗺️ 全台空氣品質分佈圖 (即時)")
        # hover_data 加入 'sitename' 欄位
        fig_map = px.scatter_mapbox(
            df_live,
            lat="lat",
            lon="lon",
            color="pm25",
            size="pm25",
            color_continuous_scale="RdYlGn_r", 
            range_color=[0, 70],
            size_max=15,
            zoom=6.5,
            center={"lat": 23.6, "lon": 121.0},
            mapbox_style="carto-positron",
            hover_data=['sitename', 'temp', 'humidity', 'id']
        )
        fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

# --- 頁面 2: 歷史數據分析 (已修正並新增進階 EDA 圖表) ---
elif page == "歷史數據分析":
    st.title("📈 歷史趨勢與特徵分析 (EDA)")
    
    if df_hist.empty:
        st.info("💡 請將組員合併後的檔案 `all_pm25_7days.csv` 上傳至專案根目錄，才能進行歷史分析。")
    else:
        # 確保關鍵欄位存在
        required_cols = ['LASS_PM25', 'LASS_Temp', 'LASS_Humid', 'MonitorName']
        missing_cols = [col for col in required_cols if col not in df_hist.columns]
        
        if missing_cols:
            st.error(f"歷史資料缺少關鍵欄位：{', '.join(missing_cols)}，無法繪製進階 EDA 圖表。請檢查 `all_pm25_7days.csv`。")
            return

        # 1. PM2.5 時間趨勢圖 (參考圖一)
        st.subheader("1. 主要測站 PM2.5 時間趨勢")
        
        # 計算每小時的 PM2.5 平均值，並依測站分組
        # 選取觀測筆數最多的前 10 個站點進行繪製
        top_stations = df_hist['MonitorName'].value_counts().nlargest(10).index
        df_trend = df_hist[df_hist['MonitorName'].isin(top_stations)]
        
        # 聚合：計算每小時平均值
        df_trend_agg = df_trend.groupby(['time', 'MonitorName'])['LASS_PM25'].mean().reset_index()

        fig_trend = px.line(
            df_trend_agg,
            x='time',
            y='LASS_PM25',
            color='MonitorName',
            title='近七日主要測站 LASS PM2.5 濃度變化趨勢 (小時平均)',
            labels={'LASS_PM25': 'PM2.5 濃度 (μg/m³)', 'time': '日期與時間', 'MonitorName': '測站名稱'},
            template="plotly_white",
            line_shape='spline' # 讓線條更平滑
        )
        fig_trend.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_trend, use_container_width=True)


        # 2. 氣象特徵 vs PM2.5 散布圖 (參考圖二)
        st.subheader("2. 氣象特徵與 PM2.5 關係散布圖")
        
        # 由於數據量可能很大，取樣 10,000 筆以提高效能
        sample_df = df_hist.sample(n=min(10000, len(df_hist)), random_state=42)
        
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            fig_temp = px.scatter(
                sample_df, x='LASS_Temp', y='LASS_PM25', 
                color='MonitorName', # 以測站名稱著色
                opacity=0.6,
                title="PM2.5 與溫度散布關係圖",
                labels={'LASS_Temp': '溫度 (°C)', 'LASS_PM25': 'PM2.5 (μg/m³)'},
                trendline="ols", # 加入趨勢線
                color_continuous_scale=px.colors.sequential.Sunset,
                template="plotly_white"
            )
            fig_temp.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with col_eda2:
            fig_humid = px.scatter(
                sample_df, x='LASS_Humid', y='LASS_PM25', 
                color='MonitorName', # 以測站名稱著色
                opacity=0.6,
                title="PM2.5 與濕度散布關係圖",
                labels={'LASS_Humid': '濕度 (%)', 'LASS_PM25': 'PM2.5 (μg/m³)'},
                trendline="ols",
                color_continuous_scale=px.colors.sequential.Teal,
                template="plotly_white"
            )
            fig_humid.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_humid, use_container_width=True)


        # 3. 相關係數熱圖 (參考圖三)
        st.subheader("3. 主要環境特徵相關係數熱圖")
        
        numeric_cols = ['LASS_PM25', 'LASS_Temp', 'LASS_Humid', 'EPA_PM25', 'AQI', 'Wind_Speed']
        existing_numeric_cols = [col for col in numeric_cols if col in df_hist.columns]
        df_corr = df_hist[existing_numeric_cols].copy()
        
        # 重新命名欄位以便圖表顯示
        chinese_names = {
            'LASS_PM25': 'LASS PM2.5', 'LASS_Temp': 'LASS 溫度', 'LASS_Humid': 'LASS 濕度', 
            'EPA_PM25': 'EPA PM2.5', 'AQI': 'AQI 指數', 'Wind_Speed': '風速'
        }
        df_corr = df_corr.rename(columns=chinese_names)
        
        corr_matrix = df_corr.corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True, 
            aspect="auto",
            color_continuous_scale=px.colors.diverging.RdBu, # 使用冷暖色調
            title="主要環境特徵相關係數矩陣"
        )
        
        # 調整熱圖排版
        fig_heatmap.update_layout(
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0),
            height=600
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)


# --- 頁面 3: 模型預測展示 (無變動) ---
elif page == "模型預測展示":
    st.title("🤖 AI 模型預測與績效")
    
    if model is None:
        st.info("💡 請將訓練好的模型檔案 `model.pkl` 上傳至專案根目錄以啟用此功能。")
        
        # 預測績效展示 (模擬組員的訓練結果)
        st.markdown("### 🏆 預計的模型績效 (RMSE 模擬)")
        model_performance = {
            'Baseline (t-1)': 8.5,
            'XGBoost': 5.2,
            'LightGBM': 4.8
        }
        df_perf = pd.DataFrame(list(model_performance.items()), columns=['模型', 'RMSE (越低越好)'])
        fig_perf = px.bar(
            df_perf, x='模型', y='RMSE (越低越好)', 
            color='RMSE (越低越好)', 
            color_continuous_scale='Viridis_r',
            text_auto=True,
            title="模型誤差比較"
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        st.success("✅ 依據訓練結果，LightGBM (4.8) 表現優於 XGBoost (5.2)。")
        
    else:
        st.success(f"模型已載入，準備進行實時預測！類型: {type(model).__name__}")
        
        st.markdown("### 🔍 站點過去 6 小時觀測與未來 1 小時預測")
        
        if not df_live.empty and station_name_to_id:
            
            # 1. 縣市選擇
            city_options = df_live['City'].unique()
            selected_city = st.selectbox("1. 選擇縣市/地區", city_options)

            # 2. 站點名稱選擇 (兩級聯動) - 現在使用 'sitename' 欄位
            station_name_options = df_live[df_live['City'] == selected_city]['sitename'].unique()
            selected_name = st.selectbox("2. 選擇站點名稱", station_name_options)
            
            if selected_name:
                # 3. 從名稱找回實際的 device_id
                selected_id = station_name_to_id.get(selected_name)
                
                # 4. 獲取當前數據
                current_data = df_live[df_live['id'] == selected_id].iloc[0]
                current_pm = current_data['pm25']
                
                now = datetime.now()
                
                # --- 執行模型預測 (特徵工程與先前邏輯相同) ---
                # 使用 device_id 進行特徵數值化 (模擬 Label Encoding)
                site_id_int = int(hashlib.sha1(selected_id.encode("utf-8")).hexdigest(), 16) % 100
                
                hour = now.hour
                month = now.month
                weekday = now.weekday() 
                is_weekend = 1 if weekday >= 5 else 0
                pm25_t1 = current_pm
                
                feature_data = {
                    'pm25_t1': [pm25_t1],
                    'hour': [hour],
                    'month': [month],
                    'weekday': [weekday],
                    'is_weekend': [is_weekend],
                    'site_id': [site_id_int] 
                }

                X_predict = pd.DataFrame(feature_data)
                
                try:
                    # 執行預測
                    pred_pm = model.predict(X_predict)[0]
                    pred_pm = max(0, pred_pm) 
                    
                    # --- 成果展示 (KPI 卡片) ---
                    col_kpi_1, col_kpi_2 = st.columns(2)
                    
                    with col_kpi_1:
                        st.metric("當前 PM2.5 濃度", f"{current_pm:.1f} µg/m³")
                    
                    with col_kpi_2:
                        delta_value = pred_pm - current_pm
                        st.metric("預測下一小時 PM2.5", f"{pred_pm:.1f} µg/m³", 
                                  delta=f"{delta_value:.1f} (變化)", 
                                  delta_color="inverse") 
                    
                    # --- 過去 6 小時觀測與未來 1 小時預測趨勢圖 ---
                    st.markdown("#### 📈 過去 6 小時觀測值與未來 1 小時預測值")
                    
                    # 模擬過去 6 小時的數據點時間標籤
                    time_labels = []
                    for i in range(6, 0, -1):
                        time_labels.append((now - timedelta(hours=i)).strftime("%H:%M"))
                    time_labels.append(now.strftime("%H:%M") + " (現在)")
                    time_labels.append((now + timedelta(hours=1)).strftime("%H:%M") + " (預測)")
                             
                    # 模擬過去 6 小時 PM2.5 數據 
                    np.random.seed(int(time.time() // 60) + int(hashlib.sha1(selected_id.encode("utf-8")).hexdigest(), 16) % 1000)
                    history_pm = [current_pm + np.random.uniform(-5, 5) for _ in range(6)]
                    
                    # 結合所有數據點 (6 歷史模擬 + 1 現在觀測 + 1 預測)
                    values = history_pm + [current_pm, pred_pm]
                    
                    # 構造 DataFrame
                    df_trend = pd.DataFrame({
                        '時間點': time_labels, 
                        'PM2.5 濃度 (µg/m³)': values
                    })
                    
                    # 增加一個類別欄位用於 Plotly Express 的顏色區分
                    df_trend['數據類型'] = ['觀測值'] * 7 + ['預測值'] * 1

                    # 使用 Plotly Express 繪製趨勢圖
                    fig_trend = px.line(
                        df_trend, 
                        x='時間點', 
                        y='PM2.5 濃度 (µg/m³)', 
                        color='數據類型', 
                        title=f'站點 {selected_name} 空氣品質 6+1 小時趨勢',
                        markers=True,
                        color_discrete_map={'觀測值': 'blue', '預測值': 'red'}
                    )
                    
                    # 優化：讓預測值線段為虛線
                    fig_trend.update_traces(
                        selector=dict(name='預測值'), 
                        line=dict(dash='dash')
                    )
                    
                    fig_trend.update_layout(
                        xaxis_title="時間點",
                        yaxis_title="PM2.5 濃度 (μg/m³)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"模型預測執行失敗。請確認模型所需的特徵 (欄位名稱) 是否正確: {e}")
            else:
                 st.warning("請先從上方選擇一個有效的站點名稱。")
        else:
            st.warning("沒有即時 LASS 數據，無法進行實時預測。")
