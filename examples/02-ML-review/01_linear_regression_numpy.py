import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm
import os

# 나눔고딕 폰트 경로 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
else:
    print(f"Warning: Font file not found at {font_path}")
    font_prop = fm.FontProperties(family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 시각화 설정
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. 데이터 로드
print("캘리포니아 주택 가격 데이터 로딩 중...")
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# 데이터 프레임으로 변환
data = pd.DataFrame(X, columns=feature_names)
data['PRICE'] = y

print("\n데이터 정보:")
print(data.info())
print("\n데이터 샘플:")
print(data.head())
print("\n기술 통계:")
print(data.describe())

# 2. 데이터 준비
# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# 3. NumPy로 선형 회귀 모델 구현
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 가중치와 편향 초기화
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 경사 하강법
        for _ in range(self.n_iterations):
            # 예측
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 손실 계산 (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            # 그래디언트 계산
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = np.mean(y_pred - y)
            
            # 가중치와 편향 업데이트
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 4. 모델 학습
print("\n모델 학습 시작...")
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_scaled, y_train_scaled)

# 5. 모델 평가
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test_scaled, y_pred)
r2 = r2_score(y_test_scaled, y_pred)

print("\n모델 성능 평가:")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# 6. 결과 시각화
# 6.1 Cost 변화 그래프
plt.figure(figsize=(10, 6))
plt.plot(range(len(model.losses)), model.losses)
plt.xlabel('에폭', fontproperties=font_prop)
plt.ylabel('Loss (MSE)', fontproperties=font_prop)
plt.title('학습 과정에서의 Loss 변화', fontproperties=font_prop)
plt.grid(True)
plt.savefig('01_linear_regression_numpy_cost_history.png')
plt.close()

# 6.2 특성 중요도 시각화
plt.figure(figsize=(12, 6))
plt.bar(feature_names, model.weights)
plt.xticks(rotation=45, fontproperties=font_prop)
plt.xlabel('특성', fontproperties=font_prop)
plt.ylabel('가중치', fontproperties=font_prop)
plt.title('특성 중요도 (NumPy)', fontproperties=font_prop)
plt.tight_layout()
plt.savefig('01_linear_regression_numpy_feature_importance.png')
plt.close()

# 6.3 예측값 vs 실제값 산점도
plt.figure(figsize=(10, 6))
plt.scatter(y_test_scaled, y_pred, alpha=0.5)
plt.plot([y_test_scaled.min(), y_test_scaled.max()], 
         [y_test_scaled.min(), y_test_scaled.max()], 'r--', lw=2)
plt.xlabel('실제 가격', fontproperties=font_prop)
plt.ylabel('예측 가격', fontproperties=font_prop)
plt.title('예측값 vs 실제값 (NumPy)', fontproperties=font_prop)
plt.grid(True)
plt.savefig('01_linear_regression_numpy_prediction_vs_actual.png')
plt.close()

# 7. 모델 방정식 출력
print("\n선형 회귀 모델 방정식:")
equation = "가격 = "
weights = model.weights * (scaler_y.scale_[0] / scaler_X.scale_)
bias = model.bias * scaler_y.scale_[0] - np.sum(weights * scaler_X.mean_) + scaler_y.mean_[0]

equation += f"{bias:.4f}"
for name, weight in zip(feature_names, weights):
    equation += f" + {weight:.4f} × {name}"
print(equation)

# 8. 예측 예시
print("\n새로운 주택 예측 예시:")
example_house = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
example_house_scaled = scaler_X.transform(example_house)
predicted_price_scaled = model.predict(example_house_scaled)
predicted_price = scaler_y.inverse_transform(predicted_price_scaled.reshape(-1, 1))[0][0]

print("입력 특성:")
for name, value in zip(feature_names, example_house[0]):
    print(f"{name}: {value:.2f}")
print(f"\n예측 가격: ${predicted_price*100000:.2f}") 