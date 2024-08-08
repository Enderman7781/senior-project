from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
np.object = object  
np.int = int    
np.bool = bool 
np.float = float
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import csv
import os
from io import StringIO
import joblib


app = Flask(__name__)
CORS(app)
print("Current Working Directory:", os.getcwd())
# 載入多個模型

models = { 
}

def getCurrentTime():
    current_time = datetime.now()
    # 減去10分鐘
    new_time = current_time - timedelta(minutes=20)
    year = new_time.strftime('%Y')
    month = new_time.strftime('%m')
    day = new_time.strftime('%d')
    hour = new_time.strftime('%H')
    minute = new_time.strftime('%M')
    second = new_time.strftime('%S')
    
    return { 'date': f'{year}{month}{day}',
            'hour':hour,
            'minute':minute,
            'second':second }
# 爬蟲函數
def fetch_data():
    now = getCurrentTime()
    base_url = f"https://tisvcloud.freeway.gov.tw/history/TDCS/M05A/{now['date']}/{now['hour']}/"
    
    floorMinute = int(now['minute']) - (int(now['minute']) % 5)
    floorMinute = f'{floorMinute:02}'

    request_url = f"{base_url}TDCS_M05A_{now['date']}_{now['hour']}{floorMinute}00.csv"
    print(request_url)
    response = requests.get(request_url)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return []

    # 解析CSV數據
    data = []
    lines = response.text.splitlines()
    csv_reader = csv.reader(lines)
    for row in csv_reader:
        # 根據你的需求來處理每一行數據
        data.append(row)
    df = pd.DataFrame(data)
    return df

def filterData(scraped_data):
    h5_str = '05F'
    
    # Step 1: 過濾資料
    filtered_data = scraped_data[(scraped_data[1].str.contains(h5_str)) | (scraped_data[2].str.contains(h5_str))]

    # 確保 [4] 和 [5] 欄位為數值型別
    filtered_data[4] = pd.to_numeric(filtered_data[4], errors='coerce')
    filtered_data[5] = pd.to_numeric(filtered_data[5], errors='coerce')

    # Step 2: 分群並計算加總，只保留每組的第一列
    grouped_data = filtered_data.groupby(1).apply(
        lambda group: group.assign(SUM_5=group[5].sum()).iloc[0:1]
    )

    # Step 3: 更新欄位 [4] 的值
    grouped_data[4] = grouped_data.apply(
        lambda row: (row[4] * row[5] / row['SUM_5']) if row['SUM_5'] != 0 else np.nan,
        axis=1
    )

    # Step 4: 刪除欄位 [3]
    grouped_data = grouped_data.drop(columns=[3])

    # Step 5: 刪除輔助欄位並重新命名欄位
    result_data = grouped_data.drop(columns=['SUM_5'])
    result_data.columns = ['時間', '起點路段', '終點路段', '平均速度', '車流數量']
    
    # Step 6: 轉換 '時間' 欄位為日期時間格式
    result_data['時間'] = pd.to_datetime(result_data['時間'])
    
    return create_features(result_data)

def create_features(df):
    df['hour'] = df['時間'].dt.hour
    df['dayofweek'] = df['時間'].dt.dayofweek
    df['month'] = df['時間'].dt.month
    df['lag1'] = df['平均速度'].shift(1)
    df['lag2'] = df['平均速度'].shift(2)
    df['moving_avg_3'] = df['平均速度'].rolling(window=3).mean()
    #df['high_load'] = ((df['dayofweek'] == 6) & (df['hour'] >= 15) & (df['hour'] <= 20)).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    # 新增手段A特征
    #df['高乘載'] = ((df['dayofweek'] == 6) & (df['hour'] >= 15) & (df['hour'] <= 20)).astype(int)  # 示例逻辑
    
    df['高乘載'] = 0
    df['匝道封閉'] = 0
    df['路肩開放'] = 0  
    df['車禍'] = 0

    df = df.dropna()  # 去除NaN值
    return df

def predict_new_data(new_data, encoder, scaler, model, route_dict):
    # If the input is a dictionary, convert it to a DataFrame
    if isinstance(new_data, dict):
        # Convert dictionary to DataFrame and ensure columns align correctly
        new_data = pd.DataFrame([new_data])
        print('轉換')

    # Ensure new_data is now a DataFrame
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError("new_data should be a DataFrame after conversion.")

    # Extract the route code and get the route name
    route_code = f"{new_data['起點路段'].iloc[0]} -> {new_data['終點路段'].iloc[0]}"
    route_name = route_dict.get(route_code, route_code)
    print(f"Predicting for route: {route_name}")

    # Drop '終點路段' and any non-numeric columns like '時間'
    new_data = new_data.drop(columns=['終點路段'], errors='ignore')

    # Ensure that '時間' or any other Timestamp is removed or converted
    if '時間' in new_data.columns:
        new_data = new_data.drop(columns=['時間'])

    # Extract features excluding '平均速度' and other target columns
    X_new_base = new_data.drop(columns=['平均速度'], errors='ignore')

    # Ensure 'X_new_base' is a DataFrame with columns
    if isinstance(X_new_base, pd.Series):
        X_new_base = X_new_base.to_frame().T

    # Encode categorical features
    X_new_encoded = encoder.transform(X_new_base[['起點路段']])

    # Remove the original categorical feature and concatenate the encoded data
    X_new_base = X_new_base.drop(columns=['起點路段'])
    X_new_base = np.hstack((X_new_base.values, X_new_encoded))

    # Convert all data to float type, ensuring that inputs are numeric
    X_new_base = np.array(X_new_base, dtype=float)

    # Standardize features
    X_new_scaled = scaler.transform(X_new_base)

    # Reshape to fit LSTM input
    X_new_reshaped = np.reshape(X_new_scaled, (X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

    # Make predictions
    y_pred_new = model.predict(X_new_reshaped)

    return y_pred_new


        
@app.route('/predict', methods=['POST'])
def predict():
    
    # 爬取數據
    scraped_data = fetch_data()
    if scraped_data.empty:
        return jsonify({'error': 'Failed to fetch data from URL'}), 500
    # 留下可用資料
    
    useful_data = filterData(scraped_data=scraped_data)
    model_to_gate = {
        ('03F0201N','05F0000S'):'南港(3)_to_南港系統',
        ('03F0158S','05F0000S'):'木柵and南深路(3)_to_南港系統',
        ('05F0000S','05F0055S'):'南港系統_to_石碇',
        ('05F0055S','05F0287S'):'石碇_to_坪林',
        ('05F0287S','05F0309S'):'坪林_to_頭城',
        ('05F0309S','05F0439S'):'頭城_to_宜蘭北',
        ('05FR113S','05F0439S'):'宜蘭南_to_羅東',
        ('05F0439S','05F0494S'):'羅東_to_蘇澳',
        
        ('05F0001N','03F0150N'):'南港系統_to_南港(3)',
        ('05F0001N','03F0201S'):'南港系統_to_木柵(3)',
        ('05F0055N','05F0001N'):'石碇_to_南港系統',
        ('05F0287N','05F0055N'):'坪林_to_石碇',
        ('05F0309N','05F0287N'):'頭城_to_坪林',
        ('05F0438N','05F0309N'):'宜蘭北_to_頭城',
        ('05F0528N','05F0438N'):'羅東_to_宜蘭北',
        ('05F0438N','05FR143N'):'羅東_to_宜蘭南'
    }
    
    mapping_file = r'./route_mapping.csv'
    route_mapping = pd.read_csv(mapping_file)
    route_dict = {row['route_code']: row['route_name'] for _, row in route_mapping.iterrows()}
    
    predictions =[]
    for index, row in useful_data.iterrows():
        start_code = row['起點路段']
        end_code = row['終點路段']
        
        # 獲取對應的模型
        model_key = (start_code, end_code)
        if model_key in model_to_gate:
            model_route = model_to_gate[model_key]
            path_to_model = f'./saved_models/{model_route}/model.h5'
            model = load_model(path_to_model, custom_objects={'Custom>Adam': Adam},compile=False) 
            encoder = joblib.load(f'./saved_models/{model_route}/encoder.pkl')
            scaler = joblib.load(f'./saved_models/{model_route}/scaler.pkl')
            
            prediction = predict_new_data(new_data=row.to_dict(),model=model,encoder=encoder,scaler=scaler,route_dict=route_dict)

            
            # 儲存預測結果
            predictions.append(prediction[0])
        else:
            # 如果找不到對應的模型，儲存一個默認值（如 None 或 -1）
            predictions.append(None)

    useful_data['預測結果'] = predictions
    
    result_json = useful_data.to_json(orient='records')

    # 返回 JSON 給前端
    return jsonify(result_json)
    
    # 將數據傳給模型進行預測
    input_data = np.array(scraped_data).reshape(1, -1)
    prediction = model.predict(input_data)
    result = prediction[0].tolist()
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
   
    #getModels()
    app.run(host='0.0.0.0', port=5000)
