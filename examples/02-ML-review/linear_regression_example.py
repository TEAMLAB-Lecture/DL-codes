import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 상수항 추가
X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

# 3. 선형 회귀 모델 구현 (Gradient Descent)
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.cost_history = []
        self.weights_history = []
    
    def compute_cost(self, X, y, weights):
        """비용 함수(MSE) 계산"""
        m = len(y)
        predictions = X.dot(weights)
        error = predictions - y
        cost = (1/(2*m)) * np.sum(error**2)
        return cost
    
    def gradient_descent(self, X, y):
        """경사 하강법으로 가중치 최적화"""
        m = len(y)
        n_features = X.shape[1]
        
        # 가중치 초기화
        self.weights = np.random.randn(n_features)
        
        # 초기 비용 계산
        cost = self.compute_cost(X, y, self.weights)
        self.cost_history.append(cost)
        self.weights_history.append(self.weights.copy())
        
        # 경사 하강법 반복
        for i in range(self.n_iterations):
            # 예측값 계산
            predictions = X.dot(self.weights)
            
            # 오차 계산
            error = predictions - y
            
            # 그래디언트 계산
            gradients = (1/m) * X.T.dot(error)
            
            # 가중치 업데이트
            self.weights = self.weights - self.learning_rate * gradients
            
            # 비용 계산
            cost = self.compute_cost(X, y, self.weights)
            self.cost_history.append(cost)
            
            # 가중치 히스토리 저장 (100회 반복마다)
            if i % 100 == 0:
                self.weights_history.append(self.weights.copy())
                
            # 100회 반복마다 학습 진행 상황 출력
            if i % 100 == 0:
                print(f"반복 {i}, 비용: {cost:.4f}")
    
    def fit(self, X, y):
        """모델 학습"""
        self.gradient_descent(X, y)
        return self
    
    def predict(self, X):
        """예측"""
        return X.dot(self.weights)

# 4. 모델 학습
print("\n모델 학습 시작...")
model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_b, y_train)

# 5. 모델 평가
# 학습 데이터에 대한 예측
y_train_pred = model.predict(X_train_b)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# 테스트 데이터에 대한 예측
y_test_pred = model.predict(X_test_b)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n모델 성능 평가:")
print(f"학습 데이터 MSE: {train_mse:.4f}")
print(f"학습 데이터 R²: {train_r2:.4f}")
print(f"테스트 데이터 MSE: {test_mse:.4f}")
print(f"테스트 데이터 R²: {test_r2:.4f}")

# 6. 결과 시각화
# 6.1 Cost 변화 그래프
plt.figure(figsize=(10, 6))
plt.plot(range(len(model.cost_history)), model.cost_history)
plt.xlabel('반복 횟수')
plt.ylabel('Cost (MSE)')
plt.title('학습 과정에서의 Cost 변화')
plt.grid(True)
plt.savefig('cost_history.png')
plt.close()

# 6.2 특성 중요도 시각화
plt.figure(figsize=(12, 6))
feature_importance = np.abs(model.weights[1:])  # 상수항 제외
plt.bar(feature_names, feature_importance)
plt.xticks(rotation=45)
plt.xlabel('특성')
plt.ylabel('절대 가중치')
plt.title('특성 중요도')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 6.3 예측값 vs 실제값 산점도
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 가격')
plt.ylabel('예측 가격')
plt.title('예측값 vs 실제값')
plt.grid(True)
plt.savefig('prediction_vs_actual.png')
plt.close()

# 7. 모델 방정식 출력
print("\n선형 회귀 모델 방정식:")
equation = "가격 = "
for i, (name, weight) in enumerate(zip(['상수항'] + feature_names, model.weights)):
    if i == 0:
        equation += f"{weight:.4f}"
    else:
        equation += f" + {weight:.4f} × {name}"
print(equation)

# 8. 예측 예시
print("\n새로운 주택 예측 예시:")
example_house = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
example_house_scaled = scaler.transform(example_house)
example_house_b = np.c_[np.ones((1, 1)), example_house_scaled]
predicted_price = model.predict(example_house_b)[0]

print("입력 특성:")
for name, value in zip(feature_names, example_house[0]):
    print(f"{name}: {value:.2f}")
print(f"\n예측 가격: ${predicted_price*100000:.2f}") 