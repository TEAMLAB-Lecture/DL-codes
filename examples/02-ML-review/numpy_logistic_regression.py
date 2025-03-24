import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

# 시각화 설정
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 재현성을 위한 시드 설정
np.random.seed(42)

# 데이터 로드 및 전처리
def load_and_preprocess_diabetes_data():
    """당뇨병 데이터셋 로드 및 전처리"""
    # 데이터 로드
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    # 당뇨병 진단 기준에 따라 이진 분류로 변환 (중간값 기준)
    y_binary = (y > np.median(y)).astype(int)
    
    # 특성 이름 설정
    feature_names = diabetes.feature_names
    
    # 특성 엔지니어링
    # 1. BMI와 혈압의 상호작용
    bmi_idx = np.where(feature_names == 'bmi')[0][0]
    bp_idx = np.where(feature_names == 'bp')[0][0]
    X = np.column_stack([X, X[:, bmi_idx] * X[:, bp_idx]])
    feature_names = np.append(feature_names, 'bmi_bp_interaction')
    
    # 2. BMI의 제곱
    X = np.column_stack([X, X[:, bmi_idx]**2])
    feature_names = np.append(feature_names, 'bmi_squared')
    
    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_binary, feature_names, scaler

# 로지스틱 회귀 모델 클래스
class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def sigmoid(self, z):
        """시그모이드 함수: 1 / (1 + exp(-z))"""
        # 오버플로우/언더플로우 방지를 위한 클리핑
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """가중치와 편향 초기화"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
    
    def forward(self, X):
        """순방향 전파: z = X·w + b, a = sigmoid(z)"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_true, y_pred):
        """이진 교차 엔트로피 손실 계산"""
        # 수치적 안정성을 위한 클리핑
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # 손실 계산: -[y·log(p) + (1-y)·log(1-p)]
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def compute_gradients(self, X, y_true, y_pred):
        """경사 계산: dL/dw, dL/db"""
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        db = (1/m) * np.sum(y_pred - y_true)
        return dw, db
    
    def update_parameters(self, dw, db):
        """경사 하강법을 사용한 가중치와 편향 업데이트"""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        """확률 예측"""
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        """클래스 예측"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """모델 학습"""
        # 파라미터 초기화
        n_features = X_train.shape[1]
        self.initialize_parameters(n_features)
        
        # 학습 과정
        for i in range(self.num_iterations):
            # 순방향 전파
            y_pred = self.forward(X_train)
            
            # 손실 계산
            loss = self.compute_loss(y_train, y_pred)
            self.losses.append(loss)
            
            # 경사 계산
            dw, db = self.compute_gradients(X_train, y_train, y_pred)
            
            # 파라미터 업데이트
            self.update_parameters(dw, db)
            
            # 학습/테스트 정확도 계산
            train_accuracy = np.mean((self.predict(X_train) == y_train).astype(int))
            self.train_accuracies.append(train_accuracy)
            
            if X_test is not None and y_test is not None:
                test_accuracy = np.mean((self.predict(X_test) == y_test).astype(int))
                self.test_accuracies.append(test_accuracy)
            
            # 진행 상황 출력
            if self.verbose and (i+1) % 100 == 0:
                if X_test is not None and y_test is not None:
                    print(f'에폭 {i+1}/{self.num_iterations}, '
                          f'손실: {loss:.4f}, '
                          f'학습 정확도: {train_accuracy:.4f}, '
                          f'테스트 정확도: {test_accuracy:.4f}')
                else:
                    print(f'에폭 {i+1}/{self.num_iterations}, '
                          f'손실: {loss:.4f}, '
                          f'학습 정확도: {train_accuracy:.4f}')
        
        return self

# 데이터 로드 및 전처리
X_scaled, y, feature_names, scaler = load_and_preprocess_diabetes_data()

# 데이터 확인
print("특성 목록:")
for i, name in enumerate(feature_names):
    print(f"{i+1}. {name}")

print("\n클래스 분포:")
print(pd.Series(y).value_counts(normalize=True))

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(LogisticRegression(verbose=False), X_scaled, y, cv=kf, scoring='accuracy')
print("\n교차 검증 정확도:", cv_scores)
print("평균 교차 검증 정확도:", np.mean(cv_scores))
print("교차 검증 정확도 표준편차:", np.std(cv_scores))

# 로지스틱 회귀 모델 학습
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train, X_test, y_test)

# 모델 평가
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# 기본 평가 지표
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n=== 모델 평가 결과 ===")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1 스코어: {f1:.4f}")

# ROC 커브
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 결과 시각화
# 1. 손실 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model.losses, label='손실')
plt.xlabel('에폭')
plt.ylabel('손실 (BCE)')
plt.title('학습 손실')
plt.legend()

# 2. 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(model.train_accuracies, label='학습 정확도')
plt.plot(model.test_accuracies, label='테스트 정확도')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.title('학습 및 테스트 정확도')
plt.legend()
plt.tight_layout()
plt.savefig('numpy_logistic_regression_learning_curves.png')
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
plt.savefig('numpy_logistic_regression_confusion_matrix.png')
plt.close()

# 4. ROC 커브
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 커브 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('numpy_logistic_regression_roc_curve.png')
plt.close()

# 5. 가중치 시각화
plt.figure(figsize=(12, 6))
colors = ['blue' if w > 0 else 'red' for w in model.weights]
plt.bar(feature_names, model.weights, color=colors)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('특성')
plt.ylabel('가중치')
plt.title('로지스틱 회귀 모델의 학습된 가중치')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('numpy_logistic_regression_weights.png')
plt.close()

# 6. 결정 경계 시각화 (PCA 사용)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

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
Z = model.predict_proba(grid_original).reshape(xx.shape)

# 결정 경계 그리기
plt.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='-')
plt.colorbar(plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdBu'))

plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.title('PCA로 시각화한 로지스틱 회귀 결정 경계')
plt.legend()
plt.tight_layout()
plt.savefig('numpy_logistic_regression_decision_boundary.png')
plt.close()

# 7. 로지스틱 회귀의 수학적 개념 시각화
plt.figure(figsize=(15, 5))

# 1. 시그모이드 함수
plt.subplot(1, 3, 1)
z = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-z))
plt.plot(z, sigmoid)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
plt.title('시그모이드 함수')
plt.xlabel('z = w·x + b')
plt.ylabel('σ(z) = 1 / (1 + e^(-z))')
plt.grid(True)

# 2. 로그 오즈
plt.subplot(1, 3, 2)
p = np.linspace(0.01, 0.99, 1000)
log_odds = np.log(p / (1 - p))
plt.plot(p, log_odds)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.3)
plt.title('로그 오즈 함수')
plt.xlabel('확률 (p)')
plt.ylabel('log(p/(1-p))')
plt.grid(True)

# 3. 이진 교차 엔트로피 손실
plt.subplot(1, 3, 3)
p = np.linspace(0.01, 0.99, 1000)
bce_loss_y1 = -np.log(p)
bce_loss_y0 = -np.log(1 - p)
plt.plot(p, bce_loss_y1, label='y=1')
plt.plot(p, bce_loss_y0, label='y=0')
plt.title('이진 교차 엔트로피 손실')
plt.xlabel('예측 확률 (p)')
plt.ylabel('-log(p) or -log(1-p)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('numpy_logistic_regression_concept_visualization.png')
plt.close()

# 새로운 환자 데이터로 예측 예시
print("\n=== 새로운 환자 데이터에 대한 예측 ===")
# 임의의 새 환자 데이터 생성 (원본 특성만)
new_patients = np.array([
    # age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
    [0.5, 0.8, 2.0, 0.6, 0.7, 1.0, -0.5, 0.3, 0.4, 0.2],  # 고위험 환자
    [0.2, 0.3, 0.1, 0.2, 0.1, 0.0, 0.5, -0.1, 0.2, 0.1]   # 저위험 환자
])

# 특성 엔지니어링 적용
bmi_idx = np.where(feature_names == 'bmi')[0][0]
bp_idx = np.where(feature_names == 'bp')[0][0]
new_patients = np.column_stack([
    new_patients,
    new_patients[:, bmi_idx] * new_patients[:, bp_idx],  # BMI와 혈압의 상호작용
    new_patients[:, bmi_idx]**2  # BMI의 제곱
])

# 데이터 정규화
new_patients_scaled = scaler.transform(new_patients)

# 예측
probabilities = model.predict_proba(new_patients_scaled)
predictions = model.predict(new_patients_scaled)

# 결과 출력
for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
    risk_level = "고위험" if i == 0 else "저위험"
    print(f"{risk_level} 환자:")
    print(f"  당뇨병 확률: {prob:.4f}")
    print(f"  예측 클래스: {'당뇨병' if pred == 1 else '정상'}")

# 모델 해석
print("\n=== 모델 해석: 특성 중요도 ===")
feature_importance = pd.DataFrame({
    '특성': feature_names,
    '가중치': model.weights,
    '절대값': np.abs(model.weights)
})
feature_importance = feature_importance.sort_values('절대값', ascending=False)
print(feature_importance)

print("\n=== 로그 오즈비 해석 ===")
print("로지스틱 회귀에서 가중치는 로그 오즈비를 나타냅니다.")
for feature, weight in zip(feature_names, model.weights):
    odds_change = np.exp(weight) - 1
    direction = "증가" if weight > 0 else "감소"
    print(f"{feature}: 1 단위 증가 시 당뇨병 오즈가 {abs(odds_change)*100:.2f}% {direction}")

# 다양한 임계값에 따른 성능 평가
thresholds = np.arange(0.1, 0.9, 0.1)
results = []
for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    results.append({
        '임계값': threshold,
        '정확도': accuracy,
        '정밀도': precision,
        '재현율': recall,
        'F1': f1
    })

print("\n=== 다양한 임계값에 따른 성능 평가 ===")
results_df = pd.DataFrame(results)
print(results_df) 