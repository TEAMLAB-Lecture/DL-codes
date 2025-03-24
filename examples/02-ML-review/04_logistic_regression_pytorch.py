import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.font_manager as fm
import seaborn as sns
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
print("유방암 데이터 로딩 중...")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names

# 데이터 프레임으로 변환
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

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
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# 3. PyTorch로 로지스틱 회귀 모델 구현
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()

# 4. 모델 학습
print("\n모델 학습 시작...")
model = LogisticRegression(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
n_epochs = 1000
losses = []

for epoch in range(n_epochs):
    # 순전파
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# 5. 모델 평가
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = accuracy_score(y_test, y_pred_class.numpy())
    cm = confusion_matrix(y_test, y_pred_class.numpy())
    fpr, tpr, _ = roc_curve(y_test, y_pred.numpy())
    roc_auc = auc(fpr, tpr)

print("\n모델 성능 평가:")
print(f"정확도: {accuracy:.4f}")
print("\n혼동 행렬:")
print(cm)

# 6. 결과 시각화
# 6.1 학습 손실 그래프
plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses)
plt.xlabel('에폭', fontproperties=font_prop)
plt.ylabel('손실 (Binary Cross Entropy)', fontproperties=font_prop)
plt.title('학습 과정에서의 손실 변화', fontproperties=font_prop)
plt.grid(True)
plt.savefig('04_logistic_regression_pytorch_loss_history.png')
plt.close()

# 6.2 ROC 곡선
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontproperties=font_prop)
plt.ylabel('True Positive Rate', fontproperties=font_prop)
plt.title('ROC Curve', fontproperties=font_prop)
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('04_logistic_regression_pytorch_roc_curve.png')
plt.close()

# 6.3 혼동 행렬
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('예측 레이블', fontproperties=font_prop)
plt.ylabel('실제 레이블', fontproperties=font_prop)
plt.title('혼동 행렬', fontproperties=font_prop)
plt.savefig('04_logistic_regression_pytorch_confusion_matrix.png')
plt.close()

# 6.4 특성 가중치
plt.figure(figsize=(12, 6))
plt.bar(feature_names, model.linear.weight.detach().numpy()[0])
plt.xticks(rotation=45, fontproperties=font_prop)
plt.xlabel('특성', fontproperties=font_prop)
plt.ylabel('가중치', fontproperties=font_prop)
plt.title('특성 가중치', fontproperties=font_prop)
plt.tight_layout()
plt.savefig('04_logistic_regression_pytorch_feature_weights.png')
plt.close()

# 7. 예측 예시
print("\n새로운 환자 예측 예시:")
example_patient = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])
example_patient_scaled = scaler.transform(example_patient)
example_patient_tensor = torch.FloatTensor(example_patient_scaled)

model.eval()
with torch.no_grad():
    prediction = model(example_patient_tensor)
    prediction_class = (prediction > 0.5).float()

print("입력 특성:")
for name, value in zip(feature_names, example_patient[0]):
    print(f"{name}: {value:.2f}")
print(f"\n예측 결과: {'악성' if prediction_class.item() == 1 else '양성'}")
print(f"악성 확률: {prediction.item():.4f}") 