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


app = Flask(__name__)
CORS(app)
print("Current Working Directory:", os.getcwd())
# 載入多個模型

model_path_south = [
    '南港(3)_to_南港系統',
    '木柵and南深路(3)_to_南港系統',
    '南港系統_to_石碇',
    '石碇_to_坪林',
    '坪林_to_頭城',
    '頭城_to_宜蘭北',
    '宜蘭南_to_羅東',
    '羅東_to_蘇澳'    
]
model_path_north = [
    '南港系統_to_南港(3)',
    '南港系統_to_木柵(3)',
    '石碇_to_南港系統',
    '坪林_to_石碇',
    '頭城_to_坪林',
    '宜蘭北_to_頭城',
    '羅東_to_宜蘭北',
    '羅東_to_宜蘭南'
]
models = { 
}

def getModels():
    for index,i in enumerate(model_path_south):
        path_to_model = f'./saved_models/{i}/model.h5'
        model_name = f'modelS{index}'
        model = load_model(path_to_model, custom_objects={'Custom>Adam': Adam},compile=False)
        models[model_name] = model
        print(path_to_model)
    for jndex,j in enumerate(model_path_north):
        path_to_model = f'./saved_models/{j}/model.h5'
        model_name = f'modelN{jndex}'
        model = load_model(path_to_model, custom_objects={'Custom>Adam': Adam},compile=False)
        models[model_name] = model
        print(path_to_model)

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
    grouped_data[4] = grouped_data.apply(lambda row: row[4] * row[5] / row['SUM_5'], axis=1)

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

@app.route('/predict', methods=['POST'])
def predict():
    
    # 爬取數據
    scraped_data = fetch_data()
    if scraped_data.empty:
        return jsonify({'error': 'Failed to fetch data from URL'}), 500
    # 留下可用資料
    
    useful_data = filterData(scraped_data=scraped_data)
    print(useful_data)
    
    
    
    useful_data = useful_data.to_dict(orient='records')
    return jsonify(useful_data)
    
    # 將數據傳給模型進行預測
    input_data = np.array(scraped_data).reshape(1, -1)
    prediction = model.predict(input_data)
    result = prediction[0].tolist()
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    #getModels()
    app.run(host='0.0.0.0', port=5000)
