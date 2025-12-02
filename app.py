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
import time # 新增 time 模組用於延遲和時間處理

# ==========================================
# 🔧 核心設定 (Person 6: 系統整合)
# ==========================================
st.set_page_config(page_title="台灣 AI 空氣品質預測戰情室", layout="wide", page_icon="🍃")

# 用於備援的測站座標 (當爬蟲失敗時使用)
STATIONS_COORDS = {
    '台北': {'lat': 25.0330, 'lon': 121.5654},
    '板橋': {'lat': 25.0129, 'lon': 121.4624},
    '桃園': {'lat': 24.9976, 'lon': 121.3033},
    '新竹': {'lat': 24.8083, 'lon': 120.9681},
    '台中': {'lat': 24.1477, 'lon': 120.6736},
    '嘉義': {'lat': 23.4800, 'lon': 120.4491},
    '台南': {'lat': 22.9902, 'lon': 120.2076},
    '高雄': {'lat': 22.6322, 'lon': 120.3013},
    '屏東': {'lat': 22.6775, 'lon': 120.4853},
    '宜蘭': {'lat': 24.7570, 'lon': 121.7584},
    '花蓮': {'lat': 23.9740, 'lon': 121.6056},
    '台東': {'lat': 22.7565, 'lon': 121.1517},
    '馬祖': {'lat': 26.1557, 'lon': 119.9577},
}

# LASS/AirBox 靜態資料源 URL
TARGET_URL = "https://pm25.lass-net.org/data/last-all-airbox.json"

# ==========================================
# 🛠️ 1. 爬蟲函數 (Person 1: 資料工程)
# ==========================================

@st.cache_data(ttl=300) # 每 5 分鐘更新一次資料
def fetch_latest_lass_data():
    """從 LASS 靜態資料源爬取最新的 PM2.5、溫濕度和地理位置資料。"""
    st.info(f"⏳ 嘗試從 LASS/AirBox 靜態資料源 ({TARGET_URL}) 獲取數據...")
    
    try:
        response = requests.get(TARGET_URL, timeout=15)
        response.raise_for_status() # 檢查 HTTP 錯誤
        
        data = response.json()
        
        if 'feeds' in data:
            records = data['feeds']
        else:
            records = data

        if not records:
            st.warning("⚠️ LASS 資料源取得成功，但無有效感測器記錄。")
            return None

        df = pd.DataFrame(records)
        
        # 關鍵欄位清理與篩選
        rename_dict = {
            's_d0': 'pm25',
            's_t0': 'temp', # 溫度
            's_h0': 'humidity', # 濕度
            'gps_lat': 'lat',
            'gps_lon': 'lon',
            'timestamp': 'time'
        }
        
        # 篩選我們需要的欄位並重新命名
        cols_to_keep = list(rename_dict.keys())
        df_clean = df[[col for col in cols_to_keep if col in df.columns]].copy()
        df_clean.rename(columns=rename_dict, inplace=True)

        # 確保必要的欄位存在
        required_cols = ['pm25', 'lat', 'lon', 'temp', 'humidity']
        for col in required_cols:
            if col not in df_clean.columns:
                df_clean[col] = np.nan # 補上缺失的欄位

        # 轉換數值型態
        for col in required_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # 過濾異常值 (台灣範圍 + 合理 PM2.5/Temp/Humidity)
        # 台灣緯度約 21-26, 經度 119-123
        df_clean = df_clean[
            (df_clean['lat'].between(21, 26)) &
            (df_clean['lon'].between(119, 123)) &
            (df_clean['pm25'].between(0, 1000))
        ].dropna(subset=['pm25', 'lat', 'lon']).reset_index(drop=True)

        st.success(f"✅ LASS 資料爬取與清理成功！取得 {len(df_clean):,} 筆有效數據。")
        return df_clean

    except requests.exceptions.RequestException as e:
        st.error(f"❌ 資料爬取失敗 (網路錯誤/超時): {e}")
        return None
    except Exception as e:
        st.error(f"❌ 資料處理失敗: {e}")
        return None

# ==========================================
# ⚙️ 2. 資料處理與模型預測 (Data Processing & Prediction)
# ==========================================

# 建立特徵工程，與訓練時保持一致
def create_features(df, station_name, current_time):
    """
    對單一小時的 LASS 數據進行特徵工程，以匹配訓練模型時的輸入。
    假設輸入的 df 已經是清理後的 LASS 即時數據。
    
    Args:
        df (pd.DataFrame): 包含 'pm25', 'temp', 'humidity', 'lat', 'lon' 的即時 LASS 數據。
        station_name (str): 預測目標測站名稱 (例如 '台北')
        current_time (datetime): 當前時間 (用於提取時間特徵)
        
    Returns:
        pd.DataFrame: 包含所有必要特徵的單行 DataFrame, 供模型預測使用。
    """
    
    # 1. 計算 LASS 數據的空間平均值作為主要輸入
    # 因為 LASS 數據點太多，我們取平均來代表當前區域的背景值
    avg_pm25 = df['pm25'].mean()
    avg_temp = df['temp'].mean()
    avg_humid = df['humidity'].mean()
    
    # 2. 獲取測站座標
    coords = STATIONS_COORDS.get(station_name)
    if not coords:
        # Fallback for unknown stations (雖然在 App 中不會發生)
        coords = {'lat': df['lat'].mean(), 'lon': df['lon'].mean()} 

    # 3. 構造特徵 DataFrame
    features = {
        'pm25_t0': avg_pm25,         # t=0 時刻的 PM2.5
        'temp_t0': avg_temp,         # t=0 時刻的溫度
        'humid_t0': avg_humid,       # t=0 時刻的濕度
        
        # 測站的地理位置
        'Station_lat': coords['lat'],
        'Station_lon': coords['lon'],
        
        # 時間特徵 (從 current_time + 1H 提取)
        'target_hour': (current_time + timedelta(hours=1)).hour,
        'target_dayofweek': (current_time + timedelta(hours=1)).weekday(),
        'target_is_weekend': (current_time + timedelta(hours=1)).weekday() >= 5,
        
        # 假設前一小時數據與當前小時數據相同 (這是時間序列預測的常見簡化)
        'pm25_t1': avg_pm25, 
        'temp_t1': avg_temp,
        'humid_t1': avg_humid,
        # 假設前兩小時數據與當前小時數據相同
        'pm25_t2': avg_pm25, 
    }
    
    # 轉換為 DataFrame (單行)
    X = pd.DataFrame([features])
    
    # 確保 bool 欄位為 int (與 LightGBM 兼容)
    X['target_is_weekend'] = X['target_is_weekend'].astype(int)
    
    # 必須確保特徵欄位的順序與訓練時一致 (LightGBM 比較不敏感，但這是好習慣)
    # 這裡我們只使用我們構造的特徵，因此順序大致固定。
    
    return X


def predict_pm25_plus_1h(model, df_latest, selected_station):
    """
    使用模型預測選定測站下一小時 (t+1) 的 PM2.5。
    
    Args:
        model: 載入的 LightGBM 模型。
        df_latest (pd.DataFrame): 最新的 LASS/AirBox 數據。
        selected_station (str): 選定的測站名稱。
        
    Returns:
        tuple: (current_pm, predicted_pm)
    """
    
    # 1. 確定當前時間 (用於特徵提取)
    # LASS 數據沒有統一的時間，我們使用系統時間作為 t=0
    current_time = datetime.now() 
    
    # 2. 構造模型特徵
    try:
        X_predict = create_features(df_latest, selected_station, current_time)
    except Exception as e:
        st.error(f"❌ 特徵工程失敗: {e}")
        # 返回一個合理的當前值和一個錯誤預測值
        return df_latest['pm25'].mean(), np.nan 

    # 3. 進行預測
    try:
        prediction = model.predict(X_predict)[0]
        # PM2.5 數值不能是負數
        predicted_pm = max(0, prediction) 
    except Exception as e:
        st.error(f"❌ 模型預測失敗: {e}")
        return df_latest['pm25'].mean(), np.nan 

    # 4. 提取當前 PM2.5 (使用 LASS 數據平均值)
    current_pm = df_latest['pm25'].mean()
    
    return current_pm, predicted_pm


# ==========================================
# 🚀 3. Streamlit App 主體
# ==========================================

def run_app():
    # 標題
    st.title("🇹🇼 台灣 AI 空氣品質預測戰情室")
    st.markdown("---")

    # 側邊欄設定
    st.sidebar.title("⚙️ 設定選單")
    station_options = list(STATIONS_COORDS.keys())
    
    # 選擇測站
    selected_station = st.sidebar.selectbox(
        "選擇預測測站 (影響地理特徵)",
        options=station_options,
        index=station_options.index('臺中') if '臺中' in station_options else 0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **數據來源:** LASS/AirBox 感測器網路 (即時數據)  
        **AI 模型:** LightGBM  
        **預測目標:** 選定測站下一小時 (t+1) PM2.5
        """
    )
    st.sidebar.markdown("---")
    
    # 爬取資料
    with st.spinner(f"⏳ 正在爬取即時空氣品質資料 ({datetime.now().strftime('%H:%M:%S')})..."):
        latest_data = fetch_latest_lass_data()
        
    if latest_data is None:
        st.error("❌ 無法取得最新的 LASS/AirBox 資料。應用程式無法運行。請檢查網路或稍後重試。")
        st.stop()
        
    # 載入模型
    model_path = 'best_lgb_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"❌ 找不到模型檔案: {model_path}。請先執行訓練腳本並將 'best_lgb_model.joblib' 儲存到根目錄。")
        # 這裡的 return 是在 run_app() 函數內，是正確的用法。
        return 

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ 模型載入失敗: {e}")
        # 這裡的 return 是在 run_app() 函數內，是正確的用法。
        return
    
    # 執行預測
    with st.spinner("🧠 正在使用 AI 模型進行預測..."):
        time.sleep(1) # 模擬預測所需時間
        current_pm, pred_pm = predict_pm25_plus_1h(model, latest_data, selected_station)

    # ------------------------------------------
    # 4. 主頁面佈局
    # ------------------------------------------
    
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.markdown(f"#### 🎯 預測目標: {selected_station}")
        st.metric(
            label="當前區域 LASS 感測器平均 PM2.5 (µg/m³)", 
            value=f"{current_pm:.1f}" if not np.isnan(current_pm) else "N/A",
            delta_color="off"
        )
        
    with col2:
        st.markdown("#### 🔮 AI 預測 (下一小時)")
        if not np.isnan(pred_pm):
            delta_value = pred_pm - current_pm
            st.metric(
                label="PM2.5 預測值 (µg/m³)",
                value=f"{pred_pm:.1f}",
                delta=f"{delta_value:.1f}",
                delta_color="inverse" # 紅色代表上升 (惡化)，綠色代表下降 (改善)
            )
        else:
             st.metric(label="PM2.5 預測值 (µg/m³)", value="預測失敗", delta="N/A", delta_color="off")


    # 狀態儀表板 (使用 HTML/Markdown 強化視覺效果)
    with col3:
        st.markdown("#### 📊 視覺化戰情指標")
        
        # 顏色和指標判斷
        if np.isnan(pred_pm):
            status_text = "預測結果錯誤"
            color_code = "#808080" # 灰色
        elif pred_pm <= 15.4:
            status_text = "優良 (Good)"
            color_code = "#09ab3b" # 綠色
        elif pred_pm <= 35.4:
            status_text = "普通 (Moderate)"
            color_code = "#0068c9" # 藍色
        elif pred_pm <= 54.4:
            status_text = "對敏感族群不健康 (Unhealthy for Sensitive Groups)"
            color_code = "#ffa400" # 橘色
        else:
            status_text = "不健康 (Unhealthy)"
            color_code = "#ff2b2b" # 紅色
            
        # 構造 HTML 儀表板
        st.markdown(
            f"""
            <div style="
                border: 2px solid {color_code}; 
                padding: 15px; 
                border-radius: 10px; 
                background-color: #f0f2f6;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
                <p style="font-size: 16px; margin: 0; color: #555;">AI 預測空氣品質狀態 ({selected_station} t+1H)</p>
                <h3 style="color: {color_code}; margin-top: 5px;">{status_text}</h3>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p>現在 (Current PM2.5)</p>
                        <h2 style="color: #0068c9;">{current_pm:.1f}</h2>
                    </div>
                    <div style="text-align: right;">
                        <p>預測 +1H (AI PM2.5)</p>
                        <h2 style="color: {'#ff2b2b' if pred_pm > current_pm and pred_pm > 54.4 else '#09ab3b' if pred_pm <= 35.4 else '#ffa400'};">
                            {pred_pm:.1f}
                            <span style="font-size:16px">
                            {'⬆' if pred_pm > current_pm else '⬇'}
                            </span>
                        </h2>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 繪製趨勢圖
    st.markdown("#### 📈 區域 PM2.5 趨勢概覽")

    # 構造數據 (基於 LASS 均值和預測值)
    times = ["-3H", "-2H", "-1H", "現在", "+1H (AI 預測)"]
    
    # 模擬過去數據波動 (簡化處理，實際應用應載入歷史數據)
    # 假設過去三小時 PM2.5 在 +/- 5 內波動
    history = [current_pm + np.random.uniform(-5, 5) for _ in range(3)] 
    # 確保模擬值不為負
    history = [max(0, x) for x in history]

    values = history + [current_pm, pred_pm]
    # 設置顏色：過去灰色，現在藍色，預測紅色
    colors = ['#808080']*3 + ['#0068c9', '#ff2b2b']
    
    # 創建數據 DataFrame
    trend_df = pd.DataFrame({
        '時間': times,
        'PM2.5 值': values,
        '類型': ['歷史']*3 + ['當前', '預測'],
        '顏色': colors
    })

    # 繪製 Plotly 散點/線圖
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df['時間'], 
        y=trend_df['PM2.5 值'], 
        mode='lines+markers',
        line=dict(color='#333333', width=2),
        marker=dict(
            size=10,
            color=trend_df['顏色'],
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f}<extra></extra>',
        name='PM2.5 趨勢'
    ))

    # 增加 PM2.5 等級水平線
    fig.add_hline(y=15.5, line_dash="dash", line_color="green", annotation_text="優良/普通界線 (15.5)")
    fig.add_hline(y=35.5, line_dash="dash", line_color="blue", annotation_text="普通/敏感族群界線 (35.5)")
    fig.add_hline(y=54.5, line_dash="dash", line_color="orange", annotation_text="敏感族群/不健康界線 (54.5)")


    fig.update_layout(
        title_text='未來一小時 PM2.5 預測與歷史趨勢',
        xaxis_title="時間",
        yaxis_title="PM2.5 (µg/m³)",
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # ------------------------------------------
    # 5. 地圖視覺化 (LASS 數據點)
    # ------------------------------------------
    st.markdown("#### 📍 LASS/AirBox 即時數據分佈 (台灣地區)")

    if not latest_data.empty:
        # 使用最新的 LASS 數據創建地圖
        map_center = [latest_data['lat'].mean(), latest_data['lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=8, tiles="cartodbpositron")

        # 根據 PM2.5 值定義顏色
        def get_pm25_color(pm):
            if pm <= 15.4: return 'green'
            if pm <= 35.4: return 'blue'
            if pm <= 54.4: return 'orange'
            return 'red'

        # 將數據點添加到地圖
        for idx, row in latest_data.iterrows():
            pm_color = get_pm25_color(row['pm25'])
            popup_html = f"""
            <b>PM2.5: {row['pm25']:.1f}</b> µg/m³<br>
            溫度: {row['temp']:.1f} °C<br>
            濕度: {row['humidity']:.1f} %
            """
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color=pm_color,
                fill=True,
                fill_color=pm_color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        # 標記選定的預測測站
        station_coords = STATIONS_COORDS.get(selected_station)
        if station_coords:
            folium.Marker(
                location=[station_coords['lat'], station_coords['lon']],
                popup=f"🎯 **AI 預測目標:** {selected_station}<br>預測 PM2.5: {pred_pm:.1f}",
                icon=folium.Icon(color='purple', icon='star')
            ).add_to(m)


        # 將地圖顯示在 Streamlit 中
        st_folium(m, width=700, height=500, key="lass_map")
    else:
        st.warning("⚠️ 沒有足夠的 LASS 數據來繪製地圖。")


# ==========================================
# 4. 程式進入點
# ==========================================
if __name__ == '__main__':
    run_app()
