from pyspark.sql.functions import *
from pyspark.sql.types import *
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 스키마 정의
schema = StructType() \
.add('date', TimestampType(), True) \
.ad('close', DoubleType(), True)

# 데이터프레임 생성
data = spark.read.option('sep', ',').schema(schema).csv('/sparkdata/bitcoin/')

# 데이터프레임 연산
data = data.select('*').where('close > 0')

# Pandas 데이터프레임으로 변환
btc = data.toPandas()

# 학습 데이터 생성
train = btc.head(len(btc)-1)

# 가격 데이터 추출
close = train['close']
close = close.values

# 시간 변수 지정
time = []
j = 1
for j in range(len(train)):
  time.append([int(j + 1)])
  j = j + 1
  
# SVR 알고리즘 모델 설
rbf_svr = SVR(kernel='rbf', C=10000, gamma=0.05)
rbf_svr.fit(time, close)

# matplotlib 한글폰트 전역 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.size'] = 14.
plt.rcParams['axes.titlesize'] = 24.
plt.rcParams['axes.labelsize'] = 18.

# 플롯 스타일 설정
plt.style.use('fivethirtyeight')

# x축 지정
x_plot = np.arange(1243)

# y축 정규화
y_plot = rbf.svr.predict(x_plot.reshape((-1, 1)))

# 플롯 구성
plt.figure(figsize=(20,10))
plt.gca().invert_xaxis()
plt.xlabel('날짜')
plt.ylabel('단위: $')

# 예측 데이터 그래프 시각화
plt.plot(x_plot, btc['close'], color='red', label='실제 비트코인 가격')
plt.plot(x_plot, y_plot, color='limegreen', label='예측 비트코인 가격')
plt.legend()

plt.show()
