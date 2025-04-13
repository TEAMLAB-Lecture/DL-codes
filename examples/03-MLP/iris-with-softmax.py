# Import necessary libraries
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import itertools

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Set numpy print options for better matrix display later
np.set_printoptions(precision=2)

# --- Data Loading and Exploration ---

# Load the digits dataset
print("Loading digits dataset...")
digit_dataset = datasets.load_digits()
print("Dataset loaded.")

# Explore the dataset keys
print("\nDataset keys:")
print(digit_dataset.keys())

# Print the dataset description
print("\nDataset Description:")
# print(digit_dataset["DESCR"]) # Description can be long, uncomment to view

# Show the first data sample reshaped as an 8x8 image
print("\nFirst data sample (reshaped 8x8):")
print(digit_dataset["data"][0].reshape(-1,8))

# Show the shape of the data array
print("\nShape of data array (samples, features):")
print(digit_dataset["data"].shape)

# Show the target names (the actual digits)
print("\nTarget names (classes):")
print(digit_dataset["target_names"])

# Show the first image data (already in 8x8 format)
print("\nFirst image data (8x8 matrix):")
print(digit_dataset["images"][0])

# Display the 8th image (index 7) using matplotlib
print("\nDisplaying an example digit image (index 7)...")
plt.figure(figsize=(3, 3)) # Create a new figure
plt.imshow(digit_dataset["images"][7], cmap=plt.cm.gray_r) # Use grayscale reversed
plt.title("Digit Image Example (Index 7)")
# plt.show() # Show plot immediately - often better to show all plots at the end

# --- Data Preparation ---

# Assign features (X) and target (y)
X = digit_dataset["data"]
y = digit_dataset["target"]

# Show the first 100 target labels
print("\nFirst 100 target labels:")
print(y[:100])

# Split the data into training and testing sets
print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # Specify test_size, add random_state
print(f"Data split: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# --- Model Training and Evaluation ---

# Initialize Logistic Regression models
print("\nInitializing Logistic Regression models...")
# One-vs-Rest (OvR) strategy
logreg_ovr = LogisticRegression(multi_class="ovr", max_iter=1000, random_state=42) # Increased max_iter
# Softmax (Multinomial) strategy
logreg_softmax = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42) # Changed solver, increased max_iter

# Train the OvR model
print("\nTraining OvR model...")
logreg_ovr.fit(X_train, y_train)
print("OvR training complete.")

# Train the Softmax model
print("\nTraining Softmax model...")
logreg_softmax.fit(X_train, y_train)
print("Softmax training complete.")
# Note: Original output showed ConvergenceWarning for 'sag', common if max_iter is too low.
# Switched to 'lbfgs' and increased max_iter, which usually helps.

# --- OvR Model Evaluation ---
print("\n--- OvR Model Evaluation ---")
y_pred_ovr = logreg_ovr.predict(X_test)
y_true = y_test # y_true is the same for all evaluations on X_test

# Confusion Matrix
print("\nConfusion Matrix (OvR):")
cm_ovr = confusion_matrix(y_true, y_pred_ovr)
print(cm_ovr)

# Classification Report
print("\nClassification Report (OvR):")
print(classification_report(y_true, y_pred_ovr))

# Accuracy Score
acc_ovr = accuracy_score(y_true, y_pred_ovr)
print(f"\nAccuracy Score (OvR): {acc_ovr:.4f}")

# --- Softmax Model Evaluation ---
print("\n--- Softmax Model Evaluation ---")
y_pred_softmax = logreg_softmax.predict(X_test)

# Accuracy Score
acc_softmax = accuracy_score(y_true, y_pred_softmax)
print(f"\nAccuracy Score (Softmax): {acc_softmax:.4f}")

# Recall Score (Macro Average)
recall_softmax = recall_score(y_true, y_pred_softmax, average="macro")
print(f"Recall Score (Softmax, Macro Avg): {recall_softmax:.4f}")

# --- Cross-Validation ---
print("\n--- Cross-Validation (Accuracy) ---")
cv_folds = 5 # Reduced folds for faster execution example
print(f"Performing {cv_folds}-fold cross-validation...")

# OvR Cross-Validation
cv_scores_ovr = cross_val_score(logreg_ovr, X, y, scoring="accuracy", cv=cv_folds, n_jobs=-1) # Use n_jobs=-1 for all cores
print(f"OvR CV Accuracy Scores: {cv_scores_ovr}")
print(f"Mean OvR CV Accuracy: {cv_scores_ovr.mean():.4f}")

# Softmax Cross-Validation
cv_scores_softmax = cross_val_score(logreg_softmax, X, y, scoring="accuracy", cv=cv_folds, n_jobs=-1)
print(f"Softmax CV Accuracy Scores: {cv_scores_softmax}")
print(f"Mean Softmax CV Accuracy: {cv_scores_softmax.mean():.4f}")


# --- Other Multiclass Strategies ---
print("\n--- Evaluating other Multiclass Strategies ---")

# OneVsRestClassifier explicitly
print("\nTraining and predicting with OneVsRestClassifier...")
ovr_clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)) # Using default LR inside
ovr_clf.fit(X_train, y_train)
y_pred_ovr_explicit = ovr_clf.predict(X_test)
acc_ovr_explicit = accuracy_score(y_true, y_pred_ovr_explicit)
print(f"Accuracy Score (Explicit OvR): {acc_ovr_explicit:.4f}")
print(f"Number of estimators in OvR Classifier: {len(ovr_clf.estimators_)}")

# OneVsOneClassifier explicitly
print("\nTraining and predicting with OneVsOneClassifier...")
ovo_clf = OneVsOneClassifier(LogisticRegression(max_iter=1000, random_state=42)) # Using default LR inside
ovo_clf.fit(X_train, y_train)
y_pred_ovo = ovo_clf.predict(X_test)
acc_ovo = accuracy_score(y_true, y_pred_ovo)
print(f"Accuracy Score (OvO): {acc_ovo:.4f}")
print(f"Number of estimators in OvO Classifier: {len(ovo_clf.estimators_)}")


# --- Confusion Matrix Plotting ---

# Define the plotting function (from scikit-learn examples)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm) # Print the matrix values

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot confusion matrix for the best performing model (likely OvO based on typical results)
print("\nPlotting Confusion Matrix for OneVsOne Classifier...")
cnf_matrix_ovo = confusion_matrix(y_true, y_pred_ovo)
class_names = digit_dataset["target_names"]

# Plot non-normalized confusion matrix
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cnf_matrix_ovo, classes=class_names,
                      title='Confusion matrix (OvO), without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cnf_matrix_ovo, classes=class_names, normalize=True,
                      title='Normalized confusion matrix (OvO)')

print("\nShowing all plots...")
plt.show() # Display all generated plots at the end

print("\nScript finished.")