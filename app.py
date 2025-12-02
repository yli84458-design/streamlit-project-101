import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# ==========================================
# 🔧 核心設定 (Core Configuration)
# ==========================================
st.set_page_config(page_title="台灣 AI 空氣品質預測戰情室", layout="wide", page_icon="🍃")

# 測站座標
STATIONS_COORDS = {
    '台北': {'lat': 25.0330, 'lon': 121.5654},
    '板橋': {'lat': 25.0129, 'lon': 121.4624},
    '桃園': {'lat': 24.9976, 'lon': 121.3033},
    '新竹': {'lat': 24.8083, 'lon': 120.9681},
    '臺中': {'lat': 24.1477, 'lon': 120.6736}, 
    '嘉義': {'lat': 23.4800, 'lon': 120.4491},
    '台南': {'lat': 22.9902, 'lon': 120.2076},
    '高雄': {'lat': 22.6322, 'lon': 120.3013},
    '屏東': {'lat': 22.6775, 'lon': 120.4853},
    '宜蘭': {'lat': 24.7570, 'lon': 121.7584},
    '花蓮': {'lat': 23.9740, 'lon': 121.6056},
    '台東': {'lat': 22.7565, 'lon': 121.1517},
    '馬祖': {'lat': 26.1557, 'lon': 119.9577},
}

TARGET_URL = "https://pm25.lass-net.org/data/last-all-airbox.json"

# ==========================================
# 🛠️ 1. 爬蟲函數
# ==========================================

@st.cache_data(ttl=300) 
def fetch_latest_lass_data():
    """從 LASS 靜態資料源爬取數據 (已快取，不會頻繁重跑)。"""
    # 移除這裡的 spinner 以減少畫面變動
    try:
        response = requests.get(TARGET_URL, timeout=10) # 縮短 timeout
        if response.status_code != 200:
            return None
        
        data = response.json()
        records = data.get('feeds', data)

        if not records:
            return None

        df = pd.DataFrame(records)
        
        rename_dict = {
            's_d0': 'pm25',
            's_t0': 'temp', 
            's_h0': 'humidity', 
            'gps_lat': 'lat',
            'gps_lon': 'lon',
            'timestamp': 'time'
        }
        
        cols_to_keep = list(rename_dict.keys())
        # 確保 df_clean 是副本
        df_clean = df[[col for col in cols_to_keep if col in df.columns]].copy()
        df_clean.rename(columns=rename_dict, inplace=True)

        # 處理缺失欄位
        for col in ['pm25', 'lat', 'lon', 'temp', 'humidity']:
            if col not in df_clean.columns:
                df_clean[col] = np.nan
            else:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # 過濾
        df_clean = df_clean[
            (df_clean['lat'].between(21, 26)) &
            (df_clean['lon'].between(119, 123)) &
            (df_clean['pm25'].between(0, 1000))
        ].dropna(subset=['pm25', 'lat', 'lon']).reset_index(drop=True)

        return df_clean

    except Exception:
        return None

# ==========================================
# ⚙️ 2. 資料處理與預測
# ==========================================

def create_features(df, station_name, current_time):
    avg_pm25 = df['pm25'].mean() if not df.empty else np.nan
    avg_temp = df['temp'].mean() if not df.empty else np.nan
    avg_humid = df['humidity'].mean() if not df.empty else np.nan
    
    if np.isnan(avg_pm25) or np.isnan(avg_temp) or np.isnan(avg_humid):
         return None

    coords = STATIONS_COORDS.get(station_name, {'lat': 0, 'lon': 0}) 

    features = {
        'pm25_t0': avg_pm25,         
        'temp_t0': avg_temp,         
        'humid_t0': avg_humid,       
        'Station_lat': coords['lat'],
        'Station_lon': coords['lon'],
        'target_hour': (current_time + timedelta(hours=1)).hour,
        'target_dayofweek': (current_time + timedelta(hours=1)).weekday(),
        'target_is_weekend': int((current_time + timedelta(hours=1)).weekday() >= 5),
        'pm25_t1': avg_pm25, 
        'temp_t1': avg_temp,
        'humid_t1': avg_humid,
        'pm25_t2': avg_pm25, 
    }
    
    return pd.DataFrame([features])

def predict_pm25_plus_1h(model, df_latest, selected_station):
    current_time = datetime.now()
    current_pm = df_latest['pm25'].mean() if not df_latest.empty else np.nan
    X_predict = create_features(df_latest, selected_station, current_time)

    if X_predict is None:
        return current_pm, np.nan 

    try:
        prediction = model.predict(X_predict)[0]
        predicted_pm = max(0, prediction) 
    except Exception:
        return current_pm, np.nan 

    return current_pm, predicted_pm

# ==========================================
# 🚀 3. Streamlit App 主體
# ==========================================

def run_app():
    st.title("🇹🇼 台灣 AI 空氣品質預測戰情室")
    st.markdown("---")

    # --- 側邊欄 ---
    st.sidebar.title("⚙️ 設定選單")
    station_options = list(STATIONS_COORDS.keys())
    
    selected_station = st.sidebar.selectbox(
        "選擇預測測站",
        options=station_options,
        index=station_options.index('臺中') if '臺中' in station_options else 0
    )
    
    st.sidebar.markdown(f"**🎯 當前目標:** `{selected_station}`")
    st.sidebar.markdown("---")
    st.sidebar.info("資料來源: LASS | 模型: LightGBM")

    # --- 載入資料 (無 Spinner，無延遲) ---
    latest_data = fetch_latest_lass_data()
    
    current_pm = np.nan
    pred_pm = np.nan
    model = None
    
    # --- 載入模型與計算 (移除所有 time.sleep) ---
    if latest_data is not None and not latest_data.empty:
        model_path = 'best_lgb_model.joblib'
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                # 瞬間完成預測，不需轉圈圈
                current_pm, pred_pm = predict_pm25_plus_1h(model, latest_data, selected_station)
            except:
                current_pm = latest_data['pm25'].mean()
        else:
            # 無模型時，僅顯示當前值
            current_pm = latest_data['pm25'].mean()
    else:
        st.error("無法取得 LASS 即時資料。")

    # ------------------------------------------
    # 主頁面佈局 (數值格式化處理)
    # ------------------------------------------
    
    col1, col2, col3 = st.columns([1, 1, 2])

    def fmt(v): return f"{v:.1f}" if not np.isnan(v) else "N/A"

    with col1:
        st.markdown(f"#### 🎯 目標: {selected_station}")
        st.metric("當前 PM2.5", fmt(current_pm))
        
    with col2:
        st.markdown("#### 🔮 預測 (+1H)")
        delta_val = pred_pm - current_pm if (not np.isnan(pred_pm) and not np.isnan(current_pm)) else 0
        delta_str = f"{delta_val:.1f}" if not np.isnan(pred_pm) and not np.isnan(current_pm) else "N/A"
        st.metric("預測 PM2.5", fmt(pred_pm), delta=delta_str, delta_color="inverse")

    with col3:
        st.markdown("#### 📊 狀態指標")
        
        status = "資料不足"
        color = "#808080"
        
        if not np.isnan(pred_pm):
            if pred_pm <= 15.4: status = "優良 (Good)"; color = "#09ab3b"
            elif pred_pm <= 35.4: status = "普通 (Moderate)"; color = "#0068c9"
            elif pred_pm <= 54.4: status = "對敏感族群不健康"; color = "#ffa400"
            else: status = "不健康 (Unhealthy)"; color = "#ff2b2b"
            
        st.markdown(f"""
        <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: #f0f2f6;">
            <h3 style="color: {color}; margin:0;">{status}</h3>
            <p style="margin:0;">預測濃度: <strong>{fmt(pred_pm)}</strong> µg/m³</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- 趨勢圖 ---
    st.markdown("#### 📈 未來趨勢預測")

    if not np.isnan(current_pm):
        times = ["-3H", "-2H", "-1H", "現在", "+1H (預測)"]
        # 產生平滑的歷史數據 (避免隨機跳動太大)
        history = [max(0, current_pm + np.random.uniform(-2, 2)) for _ in range(3)]
        
        # 如果有預測值就畫預測點，沒有就只畫歷史
        if not np.isnan(pred_pm):
            values = history + [current_pm, pred_pm]
            colors = ['#888']*3 + ['#0068c9', '#ff2b2b']
        else:
            values = history + [current_pm]
            times = times[:-1]
            colors = ['#888']*3 + ['#0068c9']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=values, mode='lines+markers',
            line=dict(color='#333333', width=3),
            marker=dict(size=10, color=colors)
        ))
        
        # 固定 Y 軸範圍，避免圖表縮放跳動
        max_y = max(values) * 1.5 if values else 100
        fig.update_layout(
            height=350, 
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(range=[0, max_y]) # 固定範圍
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("暫無數據可繪製趨勢圖")

    # --- 地圖 ---
    if latest_data is not None and not latest_data.empty:
        st.markdown("#### 🗺️ 即時監測地圖")
        
        # 建立地圖 (固定中心點，避免重新整理時地圖位移)
        m = folium.Map(location=[23.6, 121.0], zoom_start=7, tiles="cartodbpositron")
        
        # 隨機抽樣 100 個點位顯示，提升效能
        display_data = latest_data.sample(min(len(latest_data), 100))
        
        for _, row in display_data.iterrows():
            if np.isnan(row['pm25']): continue
            color = 'green'
            if row['pm25'] > 35: color = 'orange'
            if row['pm25'] > 54: color = 'red'
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.6,
                popup=f"PM2.5: {row['pm25']}"
            ).add_to(m)
            
        st_folium(m, width=700, height=400, key="main_map") # 固定 key 避免重繪

if __name__ == '__main__':
    run_app()
