import requests
import pandas as pd
import json
import time
import os

print("=" * 80)
print("🛰️ LASS/AirBox 資料爬蟲 (穩定版) - ⚡️已修正溫濕度欄位")
print("=" * 80)

# LASS 官方最新的靜態資料源
TARGET_URL = "https://pm25.lass-net.org/data/last-all-airbox.json"
OUTPUT_FILE = 'lass_latest_clean.csv'

print(f"📥 正在從 {TARGET_URL} 下載資料...")

def fetch_and_clean_lass_data():
    try:
        response = requests.get(TARGET_URL, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print("✓ 下載成功！正在解析資料...")

            if 'feeds' in data:
                records = data['feeds']
            else:
                records = data

            print(f"✓ 取得 {len(records)} 筆感測器資料")

            # 轉換為 DataFrame
            df = pd.DataFrame(records)

            # 4. 資料清理與篩選
            # 關鍵修正：增加溫度 (s_t0) 和濕度 (s_h0) 欄位
            # LASS 數據中，s_d0 是 PM2.5, s_t0 是溫度, s_h0 是濕度
            cols_to_keep = ['device_id', 's_d0', 's_t0', 's_h0', 'gps_lat', 'gps_lon', 'timestamp']
            
            # 過濾只保留需要的欄位
            df_clean = df[[col for col in cols_to_keep if col in df.columns]].copy()

            # 重新命名欄位以便理解
            rename_dict = {
                's_d0': 'pm25', 's_t0': 'temp', 's_h0': 'humidity',
                'gps_lat': 'lat', 'gps_lon': 'lon', 'timestamp': 'time', 'device_id': 'id'
            }
            df_clean.rename(columns=rename_dict, inplace=True)

            # 轉換數值型態 (現在包含 temp 和 humidity)
            for col in ['pm25', 'lat', 'lon', 'temp', 'humidity']:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            # 過濾掉沒有 PM2.5, 溫或濕度數值，或經緯度的資料
            df_clean = df_clean.dropna(subset=['pm25', 'temp', 'humidity', 'lat', 'lon'])

            # 簡單過濾異常值 (台灣範圍 + 合理 PM2.5/Temp/Humidity)
            df_clean = df_clean[
                (df_clean['lat'].between(21, 26)) &
                (df_clean['lon'].between(119, 123)) &
                (df_clean['pm25'].between(0, 500)) &
                (df_clean['temp'].between(-20, 50)) &
                (df_clean['humidity'].between(0, 100))
            ]

            print(f"✓ 清理後有效資料 (含溫濕度): {len(df_clean):,} 筆")
            
            # 將結果儲存為 CSV 檔案
            df_clean.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            print(f"✓ 資料已成功儲存至: {OUTPUT_FILE}")
            
        else:
            print(f"❌ 下載失敗，HTTP 狀態碼: {response.status_code}")

    except requests.exceptions.Timeout:
        print("❌ 錯誤: 連線超時，請檢查網路連線。")
    except requests.exceptions.RequestException as e:
        print(f"❌ 錯誤: 發生請求錯誤: {e}")
    except Exception as e:
        print(f"❌ 錯誤: 發生未預期的錯誤: {e}")

if __name__ == '__main__':
    fetch_and_clean_lass_data()
