import streamlit as st

# 設定網頁標題
st.set_page_config(page_title="我的專案首頁", page_icon="🏠")

# 側邊欄 (Sidebar)
with st.sidebar:
    st.header("功能選單")
    st.write("目前還沒有功能，敬請期待！")

# 主頁面 (Main Page)
st.title("歡迎來到我們的專案！ 👋")
st.info("這是一個由 Streamlit 架設的空白專案。")

st.divider()

# 簡單的 Placeholder (佔位符)
col1, col2 = st.columns(2)

with col1:
    st.subheader("左邊區域")
    st.write("這裡未來可以放圖表。")

with col2:
    st.subheader("右邊區域")
    st.write("這裡未來可以放數據。")
