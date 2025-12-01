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
    """載入 CSV 和 GeoJSON 檔案。"""
    df_raw = pd.DataFrame()
    geojson_data = None
    
    try:
        # 載入原始 PM2.5 數據 (用於折線圖)
        df_raw = pd.read_csv('air_quality_raw.csv')
        df_raw.rename(columns={'時間': 'Timestamp', '測站名稱': 'City', 'PM2.5': 'PM25_VALUE'}, inplace=True)
        df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'])
        # st.success("數據檔案 'air_quality_raw.csv' 載入成功。") # 移除成功訊息，讓畫面更清爽
    except FileNotFoundError:
        st.error("錯誤：找不到 'air_quality_raw.csv'。請確認檔案已上傳至專案根目錄。")
    except Exception as e:
        st.error(f"載入 'air_quality_raw.csv' 時發生錯誤: {e}")
    
    try:
        # 載入 GeoJSON 數據 (用於地圖)
        with open('data/city_data.geojson', 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        # st.success("地圖檔案 'data/city_data.geojson' 載入成功。") # 移除成功訊息
    except FileNotFoundError:
        st.error("錯誤：找不到 'data/city_data.geojson'。請確認檔案已上傳至 data/ 資料夾。")
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
# 3.2 縣市預測地圖
# --------------------
def page_map():
    st.title("🗺️ 縣市預測地圖：PM2.5 濃度分佈")
    st.info("展示各縣市當前或預測的 PM2.5 濃度。顏色越深/越暖，代表污染程度越高。")

    if geojson_data is None:
        st.warning("無法繪製地圖：GeoJSON 文件載入失敗。")
        return

    # --- 模擬預測數據 (確保 City 和 GeoJSON 的 COUNTYNAME 一致) ---
    try:
        # 從 GeoJSON 中提取縣市名稱，確保與模擬數據的 City 欄位相匹配
        city_names = [feature['properties']['COUNTYNAME'] for feature in geojson_data['features']]
    except KeyError:
        st.error("GeoJSON 格式錯誤：缺少 'COUNTYNAME' 屬性。無法匹配數據。")
        return

    # 創建模擬 PM2.5 預測值 (0-80 之間)
    np.random.seed(42) # 保持結果一致
    df_map_data = pd.DataFrame({
        'City': city_names,
        'Predicted_PM25': np.random.randint(15, 80, size=len(city_names))
    })

    # --- 地圖繪製核心邏輯 ---

    # 設置地圖中心點 (台灣北部與西部的中心點，以更好地顯示這五個城市)
    # 調整 zoom_start 確保所有城市都能被看到
    m = folium.Map(location=[24.0, 120.7], zoom_start=7, tiles="CartoDB positron")

    try:
        # ***********************************************
        # 關鍵：Folium Choropleth 繪製
        # ***********************************************
        folium.Choropleth(
            geo_data=geojson_data,
            name='PM2.5 濃度分佈',
            data=df_map_data,
            columns=['City', 'Predicted_PM25'],             # 數據來源：縣市名稱和數值
            key_on='feature.properties.COUNTYNAME',         # GeoJSON 鍵：必須與數據中的 City 欄位完全匹配
            fill_color='YlOrRd',                            # 顏色方案 (從黃到紅)
            fill_opacity=0.7,
            line_opacity=0.5, # 增加邊界線透明度，讓邊界更清晰
            legend_name='預測 PM2.5 濃度 (μg/m³)',
            highlight=True,
        ).add_to(m)

        # ----------------------------------------------------
        # 移除複雜的 GeoJsonTooltip 疊加，改用 Choropleth 內建的 Tooltip
        # ----------------------------------------------------

        # 顯示地圖
        folium_static(m, width=900, height=600)
        
        # 顯示顏色圖例
        st.caption("顏色圖例 (PM2.5)：黃色 (中等) -> 紅色 (高污染)")

    except Exception as e:
        st.error(f"地圖 Choropleth 繪製失敗，請檢查 GeoJSON 鍵名 (COUNTYNAME) 與數據欄位 (City) 是否完全匹配。錯誤詳情: {e}")
        # 如果 Choropleth 失敗，我們仍然顯示一個基礎地圖
        folium_static(m, width=900, height=600)


# --------------------
# 3.3 縣市折線圖 (趨勢分析)
# --------------------
def page_line_chart():
    st.title("📊 縣市 PM2.5 歷史趨勢分析")
    st.info("選擇一個縣市，觀察其 PM2.5 歷史變化與單一測站的最新預測點。")

    # 檢查數據是否載入
    if df_raw.empty:
        st.warning("數據缺失，無法繪製圖表。")
        return

    # 側邊欄選擇器
    all_cities = df_raw['City'].unique()
    # 確保城市列表非空
    if not list(all_cities):
        st.warning("數據中找不到縣市 (City) 名稱，請檢查 'air_quality_raw.csv' 格式。")
        return
        
    selected_city = st.selectbox("選擇縣市:", all_cities)

    df_city = df_raw[df_raw['City'] == selected_city].copy()
    
    # --- 模擬下一小時的預測值 ---
    
    # 找出最新的時間戳
    latest_time = df_city['Timestamp'].max()
    next_time = latest_time + pd.Timedelta(hours=1)
    
    # 根據最新值模擬一個下一小時的預測值 (±5)
    try:
        latest_pm25 = df_city[df_city['Timestamp'] == latest_time]['PM25_VALUE'].iloc[0]
        predicted_pm25 = latest_pm25 + np.random.uniform(-5, 5)
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
