import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import plotly.express as px
import pandas as pd

# 設定網頁標題 (必須在所有 st.開頭的函式之前)
st.set_page_config(page_title="永續城市預測平台", page_icon="🌏", layout="wide")

# ===============================================
# 11/28 任務：顏色映射邏輯 (PM2.5 -> 色階)
# ===============================================
def style_function(feature):
    """根據 GeoJSON 屬性中的 'pm25' 值設定顏色。"""
    pm25_value = feature['properties'].get('pm25', 0) # 如果沒有pm25，預設為 0
    
    # 定義色階 (這是顏色映射的實作)
    if pm25_value <= 35:
        color = 'green'     # 良好
    elif pm25_value <= 70:
        color = 'yellow'    # 普通
    else:
        color = 'red'       # 警告
    
    return {
        'fillColor': color,
        'color': color,
        'weight': 1,
        'fillOpacity': 0.7
    }

# 側邊欄與選單
with st.sidebar:
    st.header("功能導覽")
    # 這裡新增了三個頁面
    page = st.radio("請選擇頁面", ["專案總覽", "縣市預測地圖", "縣市折線圖"])
    
    st.divider()
    st.write("大數據分析期末專案")
    # 確保你已經成功將 sdg11.png 和 sdg13.png 上傳到 images/ 資料夾
    try:
        st.image("images/sdg11.png", use_column_width=True)
        st.image("images/sdg13.png", use_column_width=True)
    except:
        st.caption("SDGs 圖片載入失敗，請確認檔案路徑。")


# ===============================================
# 頁面切換邏輯
# ===============================================

if page == "專案總覽":
    # --- 總覽頁面 ---
    st.title("專案總覽：永續城市與氣候行動 🏙️")
    st.info("本專案旨在透過數據分析，探討城市發展與氣候變遷的關聯。")

    st.subheader("我們關注的聯合國永續發展目標 (SDGs)")
    st.write("SDG 11: 促使城市與人類居住具包容、安全、韌性及永續性。")
    st.write("SDG 13: 完備減緩調適行動，以因應氣候變遷及其影響。")


elif page == "縣市預測地圖":
    # --- 11/28 任務：地圖頁面 ---
    st.title("縣市數據預測地圖 🗺️")
    st.write("這是 Folium 地圖框架，用於顯示縣市的 PM2.5 預測值。")
    
    # 1. GeoJSON 讀取方式 (讀取 data/city_data.geojson)
    try:
        with open("data/city_data.geojson", "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error("錯誤：找不到 data/city_data.geojson 檔案，請確認檔案已建立。")
        st.stop()
        
    # 2. 地圖初始化 (台灣中心點)
    m = folium.Map(location=[23.6, 120.9], zoom_start=7, tiles="cartodbpositron")
    
    # 3. GeoJSON 整合與顏色映射應用
    folium.GeoJson(
        geojson_data,
        name='GeoJSON Layer',
        style_function=style_function, # 應用我們定義的 style_function
        tooltip=folium.GeoJsonTooltip(fields=['city_name', 'pm25'], aliases=['城市:', 'PM2.5:'])
    ).add_to(m)

    # 顯示地圖
    st_folium(m, height=500, width=900)
    st.caption("地圖上的顏色會根據 PM2.5 數值變化，目前使用預設佔位符數據。")


elif page == "縣市折線圖":
    # --- 11/30 任務：折線圖頁面 (使用 Plotly) ---
    st.title("縣市 PM2.5 趨勢分析 📈")
    st.info("這裡將會顯示過去 6 小時的實際數據與未來 1 小時的預測值。")
    
    # 建立一個模擬數據 (Placeholder Data)
    data = {
        '時間': pd.to_datetime([f'2025-11-30 {h}:00' for h in range(10, 17)]),
        'PM2.5 數值': [35, 40, 42, 38, 36, 45, 50],
        '類型': ['實際'] * 6 + ['預測'] * 1 # 最後一個是預測
    }
    df = pd.DataFrame(data)

    city_select = st.selectbox("請選擇要分析的縣市", ["臺北市", "新北市", "桃園市", "台中市", "高雄市"])
    st.subheader(f"📍 {city_select} PM2.5 趨勢")

    # 繪製折線圖
    fig = px.line(df, 
                  x='時間', 
                  y='PM2.5 數值', 
                  color='類型', 
                  markers=True,
                  title="近 7 小時 PM2.5 變化趨勢",
                  color_discrete_map={'實際': 'blue', '預測': 'red'}) 
    
    fig.update_layout(xaxis_title="時間 (過去 6 小時 + 未來 1 小時)", yaxis_title="PM2.5 數值 (μg/m³)")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("備註：數據為模擬佔位符數據。")
