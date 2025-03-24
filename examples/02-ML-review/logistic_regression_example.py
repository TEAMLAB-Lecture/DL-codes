import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

# 시각화 설정
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 재현성을 위한 시드 설정
np.random.seed(42)
torch.manual_seed(42)

# 데이터 생성 (당뇨병 예측 데이터셋)
def generate_diabetes_data(n_samples=1000):
    """당뇨병 예측용 가상 데이터셋 생성"""
    from sklearn.datasets import load_diabetes
    
    # 실제 당뇨병 데이터셋 로드 (특성만 사용)
    real_data = load_diabetes()
    
    # 가상 데이터 생성
    np.random.seed(42)
    X = np.random.randn(n_samples, 8)  # 8개 특성
    
    # 특성 이름 설정
    feature_names = [
        'age',          # 나이
        'bmi',          # 체질량지수
        'glucose',      # 혈당
        'bp',           # 혈압
        'insulin',      # 인슐린
        'family_hist',  # 가족력
        'physical_act', # 신체활동
        'smoking'       # 흡연
    ]
    
    # 실제 관계를 반영한 타겟 변수 생성
    # 혈당, BMI, 가족력, 나이가 당뇨병에 더 큰 영향을 미침
    weights = np.array([0.3, 0.7, 1.2, 0.5, 0.6, 0.8, -0.4, 0.4])
    z = np.dot(X, weights) + np.random.normal(0, 0.5, size=n_samples)
    
    # 시그모이드 함수를 사용하여 확률로 변환
    p = 1 / (1 + np.exp(-z))
    
    # 확률에 따라 0 또는 1로 레이블 생성
    y = (np.random.random(n_samples) < p).astype(int)
    
    # 데이터프레임 생성
    df = pd.DataFrame(X, columns=feature_names)
    df['diabetes'] = y
    
    return df

# 데이터 생성
data = generate_diabetes_data(n_samples=1000)

# 데이터 확인
print("데이터셋 샘플:")
print(data.head())
print("\n기본 통계:")
print(data.describe())
print("\n클래스 분포:")
print(data['diabetes'].value_counts())

# 데이터 전처리
X = data.drop('diabetes', axis=1).values
y = data['diabetes'].values

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# 로지스틱 회귀 모델 정의
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 선형 결합: z = w*x + b
        z = self.linear(x)
        # 시그모이드 함수: P(y=1|x) = 1/(1+e^(-z))
        probability = self.sigmoid(z)
        return probability
    
    def predict(self, x, threshold=0.5):
        """주어진 임계값(기본 0.5)을 기준으로 클래스 예측"""
        with torch.no_grad():
            probabilities = self(x)
            predictions = (probabilities >= threshold).float()
            return predictions

# 입력 차원 (특성 수)
input_dim = X_train.shape[1]

# 모델 초기화
model = LogisticRegressionModel(input_dim)

# 손실 함수와 옵티마이저 설정
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습 파라미터
num_epochs = 1000
batch_size = 32
num_batches = int(np.ceil(len(X_train) / batch_size))

# 학습 이력 저장용
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# 모델 학습
print("\n학습 시작...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    # 미니배치 학습
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(X_train))
        
        # 배치 데이터 준비
        batch_X = X_train_tensor[start_idx:end_idx]
        batch_y = y_train_tensor[start_idx:end_idx]
        
        # Forward pass
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        
        # Backward pass 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * (end_idx - start_idx)
    
    # 에폭당 평균 손실 계산
    epoch_loss /= len(X_train)
    
    # 학습셋과 테스트셋에 대한 성능 평가
    model.eval()
    with torch.no_grad():
        # 학습셋 성능
        train_pred = model.predict(X_train_tensor)
        train_accuracy = (train_pred == y_train_tensor).float().mean().item()
        
        # 테스트셋 성능
        test_pred = model.predict(X_test_tensor)
        test_accuracy = (test_pred == y_test_tensor).float().mean().item()
        test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
    
    # 학습 이력 저장
    train_losses.append(epoch_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
    # 학습 진행 상황 출력 (100 에폭마다)
    if (epoch + 1) % 100 == 0:
        print(f'에폭 {epoch+1}/{num_epochs}, '
              f'손실: {epoch_loss:.4f}, '
              f'학습 정확도: {train_accuracy:.4f}, '
              f'테스트 정확도: {test_accuracy:.4f}')

print("학습 완료!")

# 모델 평가
model.eval()
with torch.no_grad():
    y_pred = model.predict(X_test_tensor).numpy().flatten()
    y_true = y_test.flatten()
    
    # 정확도, 정밀도, 재현율, F1 스코어 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)

print("\n=== 모델 평가 결과 ===")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1 스코어: {f1:.4f}")

# 결과 시각화
# 1. 손실 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='학습 손실')
plt.plot(test_losses, label='테스트 손실')
plt.xlabel('에폭')
plt.ylabel('손실 (BCE)')
plt.title('학습 및 테스트 손실')
plt.legend()

# 2. 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='학습 정확도')
plt.plot(test_accuracies, label='테스트 정확도')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.title('학습 및 테스트 정확도')
plt.legend()
plt.tight_layout()
plt.savefig('logistic_regression_learning_curves.png')
plt.close()

# 3. 혼동 행렬
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['정상', '당뇨병'], 
            yticklabels=['정상', '당뇨병'])
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title('혼동 행렬 (Confusion Matrix)')
plt.tight_layout()
plt.savefig('logistic_regression_confusion_matrix.png')
plt.close()

# 학습된 가중치와 편향 추출
with torch.no_grad():
    weights = model.linear.weight.numpy().flatten()
    bias = model.linear.bias.item()

# 4. 가중치 시각화
plt.figure(figsize=(10, 6))
feature_names = data.columns[:-1]  # 당뇨병 열 제외
colors = ['blue' if w > 0 else 'red' for w in weights]
plt.bar(feature_names, weights, color=colors)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('특성')
plt.ylabel('가중치')
plt.title('로지스틱 회귀 모델의 학습된 가중치')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('logistic_regression_weights.png')
plt.close()

# 5. 결정 경계 시각화 (2D로 차원 축소)
# PCA로 2차원으로 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 결정 경계 그리기
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='정상', alpha=0.6)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='당뇨병', alpha=0.6)

# 격자 생성
h = 0.01
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# PCA 역변환 (2D -> 원래 차원)
grid = np.c_[xx.ravel(), yy.ravel()]
grid_original = pca.inverse_transform(grid)

# 예측
with torch.no_grad():
    Z = model(torch.FloatTensor(grid_original)).numpy().reshape(xx.shape)

# 결정 경계 그리기
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')
plt.colorbar(plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu'))

plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title('PCA로 시각화한 로지스틱 회귀 결정 경계')
plt.legend()
plt.tight_layout()
plt.savefig('logistic_regression_decision_boundary.png')
plt.close()

# 새로운 환자 데이터로 예측 예시
print("\n=== 새로운 환자 데이터에 대한 예측 ===")
# 임의의 새 환자 데이터 생성
new_patients = np.array([
    # age, bmi, glucose, bp, insulin, family_hist, physical_act, smoking
    [0.5, 0.8, 2.0, 0.6, 0.7, 1.0, -0.5, 0.3],  # 고위험 환자
    [0.2, 0.3, 0.1, 0.2, 0.1, 0.0, 0.5, -0.1]   # 저위험 환자
])

# 데이터 정규화
new_patients_scaled = scaler.transform(new_patients)

# 예측
with torch.no_grad():
    probabilities = model(torch.FloatTensor(new_patients_scaled)).numpy()
    predictions = (probabilities >= 0.5).astype(int)

# 결과 출력
for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
    risk_level = "고위험" if i == 0 else "저위험"
    print(f"{risk_level} 환자:")
    print(f"  당뇨병 확률: {prob[0]:.4f}")
    print(f"  예측 클래스: {'당뇨병' if pred[0] == 1 else '정상'}")

# 모델 해석 - 각 특성의 중요도와 로그 오즈비 계산
print("\n=== 모델 해석: 특성 중요도 ===")
feature_importance = pd.DataFrame({
    '특성': feature_names,
    '가중치': weights,
    '절대값': np.abs(weights)
})
feature_importance = feature_importance.sort_values('절대값', ascending=False)
print(feature_importance)

print("\n=== 로그 오즈비 해석 ===")
print("로지스틱 회귀에서 가중치는 로그 오즈비를 나타냅니다.")
for feature, weight in zip(feature_names, weights):
    odds_change = np.exp(weight) - 1
    direction = "증가" if weight > 0 else "감소"
    print(f"{feature}: 1 단위 증가 시 당뇨병 오즈가 {abs(odds_change)*100:.2f}% {direction}") 