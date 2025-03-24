import torch
import torch.nn as nn
import torch.optim as optim
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

# 재현성을 위한 시드 설정
np.random.seed(42)
torch.manual_seed(42)

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

# PyTorch 텐서로 변환
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# 3. PyTorch로 선형 회귀 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# 모델, 손실 함수, 옵티마이저 초기화
model = LinearRegressionModel(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 모델 학습
num_epochs = 1000
cost_history = []
weight_history = []

print("\n학습 시작...")
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # 손실 값 저장
    cost_history.append(loss.item())
    
    # 가중치 저장 (100번째 에폭마다)
    if epoch % 100 == 0:
        weight_history.append([w.clone().detach().numpy() for w in model.parameters()])
        print(f'에폭: {epoch}, 손실: {loss.item():.4f}')
    
    # Backward pass와 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("학습 완료!")

# 5. 모델 평가
model.eval()
with torch.no_grad():
    # 학습 데이터에 대한 예측
    y_train_pred = model(X_train_tensor).numpy()
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # 테스트 데이터에 대한 예측
    y_test_pred = model(X_test_tensor).numpy()
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
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('에폭')
plt.ylabel('Loss (MSE)')
plt.title('학습 과정에서의 Loss 변화')
plt.grid(True)
plt.savefig('pytorch_cost_history.png')
plt.close()

# 6.2 특성 중요도 시각화
plt.figure(figsize=(12, 6))
feature_importance = np.abs(model.linear.weight.detach().numpy()[0])
plt.bar(feature_names, feature_importance)
plt.xticks(rotation=45)
plt.xlabel('특성')
plt.ylabel('절대 가중치')
plt.title('특성 중요도 (PyTorch)')
plt.tight_layout()
plt.savefig('pytorch_feature_importance.png')
plt.close()

# 6.3 예측값 vs 실제값 산점도
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 가격')
plt.ylabel('예측 가격')
plt.title('예측값 vs 실제값 (PyTorch)')
plt.grid(True)
plt.savefig('pytorch_prediction_vs_actual.png')
plt.close()

# 7. 모델 방정식 출력
print("\n선형 회귀 모델 방정식:")
equation = "가격 = "
weights = model.linear.weight.detach().numpy()[0]
bias = model.linear.bias.detach().numpy()[0]

# 원래 스케일로 변환
original_weights = weights * (y_std / X_std)
original_bias = bias * y_std - np.sum(original_weights * X_mean) + y_mean

equation += f"{original_bias:.4f}"
for name, weight in zip(feature_names, original_weights):
    equation += f" + {weight:.4f} × {name}"
print(equation)

# 8. 예측 예시
print("\n새로운 주택 예측 예시:")
example_house = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
example_house_scaled = scaler.transform(example_house)
example_house_tensor = torch.tensor(example_house_scaled, dtype=torch.float32)

with torch.no_grad():
    predicted_price = model(example_house_tensor).item()

print("입력 특성:")
for name, value in zip(feature_names, example_house[0]):
    print(f"{name}: {value:.2f}")
print(f"\n예측 가격: ${predicted_price*100000:.2f}")

# 9. PyTorch 고급 모델 구현 (nn.Sequential 사용)
print("\n고급 PyTorch 모델 구현 (nn.Sequential):")

# 모델 재정의
sequential_model = nn.Sequential(
    nn.Linear(X_train.shape[1], 1)
)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.SGD(sequential_model.parameters(), lr=0.01)

# 빠른 학습 (같은 데이터 사용)
for epoch in range(500):
    # Forward pass
    y_pred = sequential_model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # Backward pass와 최적화
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'에폭: {epoch}, 손실: {loss.item():.4f}')

print("완료!")

# 최종 가중치와 편향
with torch.no_grad():
    seq_weights = sequential_model[0].weight.detach().numpy()[0]
    seq_bias = sequential_model[0].bias.detach().numpy()[0]
    
    # 원래 스케일로 변환
    seq_original_weights = seq_weights * (y_std / X_std)
    seq_original_bias = seq_bias * y_std - np.sum(seq_original_weights * X_mean) + y_mean

print(f"\nnn.Sequential 모델 파라미터:")
print(f"편향 (w0): {seq_original_bias:.4f}")
for name, weight in zip(feature_names, seq_original_weights):
    print(f"가중치 ({name}): {weight:.4f}") 