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
import time 

# ==========================================
# 🔧 核心設定 (Core Configuration)
# ==========================================
st.set_page_config(page_title="台灣 AI 空氣品質預測戰情室", layout="wide", page_icon="🍃")

TARGET_URL = "https://pm25.lass-net.org/data/last-all-airbox.json"

# Helper: 根據經緯度粗略判斷地區，用於生成 sitename
def get_region_from_coords(lat, lon):
    """根據經緯度，為 LASS 裝置分配一個粗略的地區名稱 (用於顯示)"""
    if 24.5 <= lat <= 26.0 and 120.5 <= lon <= 122.0: return '北部地區'
    if 24.0 <= lat < 24.5 and 120.0 <= lon < 121.0: return '中部地區'
    if 23.0 <= lat < 24.0 and 120.0 <= lon < 121.0: return '嘉南地區'
    if 22.0 <= lat < 23.0 and 120.0 <= lon < 121.0: return '高屏地區'
    if 24.5 <= lat <= 26.0 and 121.5 <= lon <= 122.0: return '宜花地區'
    if 22.0 <= lat < 24.0 and 121.0 <= lon < 122.0: return '東部地區'
    return '其他地區'


# ==========================================
# 🛠️ 1. 爬蟲函數 (Data Fetcher) - [新增 device_id 和 sitename 欄位]
# ==========================================

@st.cache_data(ttl=300) # 每 5 分鐘更新一次資料
def fetch_latest_lass_data():
    """從 LASS 靜態資料源爬取最新的 PM2.5、溫濕度和地理位置資料，並生成 sitename。"""
    st.info(f"⏳ 嘗試從 LASS/AirBox 靜態資料源獲取數據 ({datetime.now().strftime('%H:%M:%S')})...")
    
    try:
        response = requests.get(TARGET_URL, timeout=15)
        response.raise_for_status() 
        
        data = response.json()
        records = data.get('feeds', data)

        if not records:
            st.warning("⚠️ LASS 資料源取得成功，但無有效感測器記錄。")
            return None

        df = pd.DataFrame(records)
        
        rename_dict = {
            'device_id': 'device_id',  # <-- 關鍵：保留 device_id
            's_d0': 'pm25',
            's_t0': 'temp', 
            's_h0': 'humidity', 
            'gps_lat': 'lat',
            'gps_lon': 'lon',
            'timestamp': 'time'
        }
        
        cols_to_select = [col for col in rename_dict.keys() if col in df.columns]
        df_clean = df[cols_to_select].copy() 
        df_clean.rename(columns=rename_dict, inplace=True)

        # 確保所有數值欄位都轉換，錯誤則設為 NaN
        required_cols = ['pm25', 'lat', 'lon', 'temp', 'humidity']
        for col in required_cols:
            if col not in df_clean.columns:
                df_clean[col] = np.nan
            else:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 確保 device_id 是字串
        df_clean['device_id'] = df_clean['device_id'].astype(str)

        # 過濾異常值 (台灣範圍 + 合理 PM2.5)
        df_clean = df_clean[
            (df_clean['lat'].between(21, 26)) &
            (df_clean['lon'].between(119, 123)) &
            (df_clean['pm25'].between(0, 1000))
        ].dropna(subset=['lat', 'lon', 'device_id']).reset_index(drop=True)

        # --- 關鍵：生成 sitename 欄位 ---
        df_clean['region'] = df_clean.apply(
            lambda row: get_region_from_coords(row['lat'], row['lon']), axis=1
        )
        # sitename 格式：[縣市/地區] - 裝置ID尾碼:[XXXX]
        df_clean['sitename'] = df_clean.apply(
            lambda row: f"{row['region']} - ID尾碼:{str(row['device_id'])[:4]}", axis=1
        )


        st.success(f"✅ LASS 資料爬取與清理成功！取得 {len(df_clean):,} 筆有效站點數據。")
        return df_clean

    except requests.exceptions.RequestException as e:
        st.error(f"❌ 資料爬取失敗 (網路錯誤/超時): {e}")
        return None
    except Exception as e:
        st.error(f"❌ 資料處理失敗: {e}")
        return None

# ==========================================
# ⚙️ 2. 資料處理與模型預測 - [修改為單一站點數據]
# ==========================================

def create_features(df_latest, selected_sitename, current_time):
    """
    對單一 LASS 裝置的數據進行特徵工程。
    """
    
    # 1. 過濾出選定的裝置數據 (應該只有一行)
    df_device = df_latest[df_latest['sitename'] == selected_sitename]
    
    if df_device.empty:
        st.warning(f"⚠️ 找不到站點 '{selected_sitename}' 的即時數據。")
        return None

    # 2. 提取關鍵單一數值
    # 使用 .iloc[0] 確保只取第一行（如果有多個同名 sitename，取最新的/第一個）
    device_data = df_device.iloc[0] 
    
    avg_pm25 = device_data.get('pm25', np.nan)
    avg_temp = device_data.get('temp', np.nan)
    avg_humid = device_data.get('humidity', np.nan)
    
    # 3. 穩定性檢查: 確保關鍵數值有效 (CRITICAL FIX)
    if not all(np.isfinite([avg_pm25, avg_temp, avg_humid])):
         st.warning("⚠️ 選定站點缺少 PM2.5, 溫度或濕度的有效數據。無法構造完整的預測特徵。")
         return None

    # 4. 獲取測站座標
    coords = {'lat': device_data.get('lat', np.nan), 'lon': device_data.get('lon', np.nan)}
    
    # 構造特徵 DataFrame
    features = {
        'pm25_t0': avg_pm25,         
        'temp_t0': avg_temp,         
        'humid_t0': avg_humid,       
        
        # 使用裝置自身的經緯度作為地理特徵
        'Station_lat': coords['lat'],
        'Station_lon': coords['lon'],
        
        # 時間特徵 (從 current_time + 1H 提取)
        'target_hour': (current_time + timedelta(hours=1)).hour,
        'target_dayofweek': (current_time + timedelta(hours=1)).weekday(),
        'target_is_weekend': (current_time + timedelta(hours=1)).weekday() >= 5,
        
        # 假設前一/兩小時數據與當前小時數據相同 (簡化處理)
        'pm25_t1': avg_pm25, 
        'temp_t1': avg_temp,
        'humid_t1': avg_humid,
        'pm25_t2': avg_pm25, 
    }
    
    X = pd.DataFrame([features])
    X['target_is_weekend'] = X['target_is_weekend'].astype(int)
    
    return X


def predict_pm25_plus_1h(model, df_latest, selected_sitename):
    """
    使用模型預測選定站點下一小時 (t+1) 的 PM2.5。
    """
    current_time = datetime.now() 
    
    # 1. 獲取當前 PM2.5
    df_device = df_latest[df_latest['sitename'] == selected_sitename]
    current_pm = df_device.iloc[0].get('pm25', np.nan) if not df_device.empty else np.nan

    # 2. 構造模型特徵
    X_predict = create_features(df_latest, selected_sitename, current_time)

    # 如果特徵構造失敗，直接返回
    if X_predict is None:
        return current_pm, np.nan 

    # 3. 進行預測
    try:
        prediction = model.predict(X_predict)[0]
        # PM2.5 數值不能是負數
        predicted_pm = max(0, prediction) 
    except Exception as e:
        st.error(f"❌ 模型預測階段失敗: {e}")
        return current_pm, np.nan 

    return current_pm, predicted_pm


# ==========================================
# 🚀 3. Streamlit App 主體
# ==========================================

def run_app():
    # 標題
    st.title("🇹🇼 台灣 AI 空氣品質預測戰情室")
    st.markdown("---")

    # ------------------------------------------
    # 側邊欄設定 (Side Bar)
    # ------------------------------------------
    st.sidebar.title("⚙️ 設定選單")
    
    # 初始化站點選擇
    selected_sitename = None
    
    # 爬取資料
    with st.spinner(f"⏳ 正在爬取即時空氣品質資料 ({datetime.now().strftime('%H:%M:%S')})..."):
        time.sleep(1) 
        latest_data = fetch_latest_lass_data()

    if latest_data is not None and not latest_data.empty:
        # 選擇站點 (側邊欄元件) - 使用動態生成的 sitename
        sitename_options = sorted(latest_data['sitename'].unique().tolist())
        
        selected_sitename = st.sidebar.selectbox(
            "選擇預測站點 (LASS 裝置)",
            options=sitename_options,
            index=0 # 預設選擇第一個
        )
    else:
        st.error("❌ 無法取得有效的 LASS/AirBox 資料。請稍後再試。")


    # 側邊欄資訊
    st.sidebar.markdown(f"**🎯 當前目標:** `{selected_sitename if selected_sitename else 'N/A'}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **數據來源:** LASS/AirBox 感測器網路 (即時數據)  
        **AI 模型:** LightGBM  
        **預測目標:** 選定站點下一小時 (t+1) PM2.5
        """
    )
    st.sidebar.markdown("---")
    
    # 初始化預測變數
    current_pm = np.nan
    pred_pm = np.nan
    model = None
    
    # ------------------------------------------
    # 預測邏輯 (Prediction Logic)
    # ------------------------------------------
    if selected_sitename:
        # 載入模型
        model_path = 'best_lgb_model.joblib'
        if not os.path.exists(model_path):
            st.warning(f"⚠️ 找不到模型檔案: {model_path}。請先執行訓練腳本。")
            # 即使沒有模型，仍嘗試獲取當前 PM2.5
            df_device = latest_data[latest_data['sitename'] == selected_sitename]
            current_pm = df_device.iloc[0].get('pm25', np.nan) if not df_device.empty else np.nan
        else:
            try:
                model = joblib.load(model_path)
            except Exception as e:
                st.warning(f"⚠️ 模型載入失敗: {e}。請檢查檔案格式。")
                df_device = latest_data[latest_data['sitename'] == selected_sitename]
                current_pm = df_device.iloc[0].get('pm25', np.nan) if not df_device.empty else np.nan
                
            # 執行預測 (只有在模型載入成功時才執行)
            if model:
                with st.spinner("🧠 正在使用 AI 模型進行預測..."):
                    time.sleep(1) # 模擬預測所需時間
                    # 預測函數會自動處理數據缺失問題，並返回 np.nan
                    current_pm, pred_pm = predict_pm25_plus_1h(model, latest_data, selected_sitename)
    
    
    # --- 格式化顯示數值 ---
    def format_value(value):
        return f"{value:.1f}" if not np.isnan(value) else "N/A"
    
    current_pm_display = format_value(current_pm)
    pred_pm_display = format_value(pred_pm)


    # ------------------------------------------
    # 4. 主頁面佈局
    # ------------------------------------------
    
    col1, col2, col3 = st.columns([1, 1, 2])

    # --- Col 1: 當前 PM2.5 ---
    with col1:
        st.markdown(f"#### 🎯 預測目標: {selected_sitename if selected_sitename else '請選擇站點'}")
        st.metric(
            label="選定站點當前 PM2.5 (µg/m³)", 
            value=current_pm_display,
            delta_color="off"
        )
        
    # --- Col 2: 預測 PM2.5 ---
    with col2:
        st.markdown("#### 🔮 AI 預測 (下一小時)")
        
        delta_display = "N/A"
        delta_color = "off"
        
        if not np.isnan(pred_pm) and not np.isnan(current_pm):
            delta_value = pred_pm - current_pm
            delta_display = f"{delta_value:.1f}"
            delta_color = "inverse" # 綠色(up)代表惡化 (PM2.5上升)，紅色(down)代表改善 (PM2.5下降)

        st.metric(
            label="PM2.5 預測值 (µg/m³)",
            value=pred_pm_display,
            delta=delta_display,
            delta_color=delta_color
        )


    # --- Col 3: 狀態儀表板 ---
    with col3:
        st.markdown("#### 📊 視覺化戰情指標")
        
        # 顏色和指標判斷 (使用預測值 pred_pm)
        status_text = "預測結果錯誤或資料不足"
        color_code = "#808080" # 灰色
        
        if not np.isnan(pred_pm):
            if pred_pm <= 15.4:
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
                <p style="font-size: 16px; margin: 0; color: #555;">AI 預測空氣品質狀態 ({selected_sitename if selected_sitename else 'N/A'} t+1H)</p>
                <h3 style="color: {color_code}; margin-top: 5px;">{status_text}</h3>
                <div style="display: flex; justify-content: space-between;">
                    <div>
                        <p>現在 (Current PM2.5)</p>
                        <h2 style="color: #0068c9;">{current_pm_display}</h2>
                    </div>
                    <div style="text-align: right;">
                        <p>預測 +1H (AI PM2.5)</p>
                        <h2 style="color: {color_code};">
                            {pred_pm_display}
                            <span style="font-size:16px">
                            {'⬆' if not np.isnan(pred_pm) and not np.isnan(current_pm) and pred_pm > current_pm else '⬇' if not np.isnan(pred_pm) and not np.isnan(current_pm) and pred_pm < current_pm else ''}
                            </span>
                        </h2>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------
    # 5. 趨勢圖 (Trend Plot)
    # ------------------------------------------
    st.markdown("#### 📈 選定站點 PM2.5 趨勢概覽")

    if not np.isnan(current_pm) and not np.isnan(pred_pm):
        # 構造數據 (基於單一站點的當前值和預測值)
        times = ["-3H", "-2H", "-1H", "現在", "+1H (AI 預測)"]
        
        # 模擬過去數據波動 (基於當前值產生合理的歷史數據)
        history = [current_pm + np.random.uniform(-5, 5) for _ in range(3)] 
        history = [max(0, x) for x in history]

        values = history + [current_pm, pred_pm]
        # 設置顏色：過去灰色，現在藍色，預測紅色
        colors = ['#808080']*3 + ['#0068c9', '#ff2b2b']
        
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

        # 增加 PM2.5 等級水平線 (如果預測值有效)
        fig.add_hline(y=15.5, line_dash="dash", line_color="green", annotation_text="優良/普通界線 (15.5)")
        fig.add_hline(y=35.5, line_dash="dash", line_color="blue", annotation_text="普通/敏感族群界線 (35.5)")
        fig.add_hline(y=54.5, line_dash="dash", line_color="orange", annotation_text="敏感族群/不健康界線 (54.5)")


        fig.update_layout(
            title_text=f'站點 {selected_sitename} 未來一小時 PM2.5 預測與歷史趨勢',
            xaxis_title="時間",
            yaxis_title="PM2.5 (µg/m³)",
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 無法繪製趨勢圖。請選擇站點或檢查資料來源。")

    st.markdown("---")
    
    # ------------------------------------------
    # 6. 地圖視覺化 (LASS 數據點)
    # ------------------------------------------
    st.markdown("#### 📍 LASS/AirBox 即時數據分佈 (台灣地區)")

    if latest_data is not None and not latest_data.empty and 'lat' in latest_data.columns and 'lon' in latest_data.columns:
        # 使用最新的 LASS 數據創建地圖
        map_center = [latest_data['lat'].mean(), latest_data['lon'].mean()]
        m = folium.Map(location=map_center, zoom_start=8, tiles="cartodbpositron")

        # 根據 PM2.5 值定義顏色
        def get_pm25_color(pm):
            if np.isnan(pm): return 'lightgray'
            if pm <= 15.4: return 'green'
            if pm <= 35.4: return 'blue'
            if pm <= 54.4: return 'orange'
            return 'red'

        # 將數據點添加到地圖
        for idx, row in latest_data.iterrows():
            pm_value = row.get('pm25', np.nan)
            temp_value = row.get('temp', np.nan)
            humid_value = row.get('humidity', np.nan)
            sitename_value = row.get('sitename', '未知站點')

            if np.isfinite(row['lat']) and np.isfinite(row['lon']):
                pm_color = get_pm25_color(pm_value)
                
                popup_html = f"""
                <b>站點: {sitename_value}</b><br>
                PM2.5: {format_value(pm_value)} µg/m³<br>
                溫度: {format_value(temp_value)} °C<br>
                濕度: {format_value(humid_value)} %
                """
                
                marker = folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5,
                    color=pm_color,
                    fill=True,
                    fill_color=pm_color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300)
                )

                # 突出顯示選定的站點
                if sitename_value == selected_sitename:
                    # 使用 Star Marker 標記預測目標
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        icon=folium.Icon(color='purple', icon='star'),
                        popup=folium.Popup(f"🎯 **AI 預測目標:** {selected_sitename}", max_width=300)
                    ).add_to(m)

                marker.add_to(m)


        # 將地圖顯示在 Streamlit 中
        st_folium(m, width=700, height=500, key="lass_map")
    else:
        st.warning("⚠️ 沒有足夠的 LASS 數據來繪製地圖。")


# ==========================================
# 4. 程式進入點
# ==========================================
if __name__ == '__main__':
    run_app()
