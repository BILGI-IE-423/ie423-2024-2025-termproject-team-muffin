import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import sys
# Ensure required folders exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("graphics", exist_ok=True)

# ========== Check if required input files exist ==========
required_files = [
    "outputs/audience/ensemble_input_DistilBERT_model.csv",
    "outputs/audience/ensemble_input_glove_model.csv",
    "outputs/audience/ensemble_input_xgboost_model.csv"
]

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print("\n❌ ERROR: The following required input files are missing:")
    for f in missing_files:
        print(f"   - {f}")
    print("\n⚠️ Please run the individual models (DistilBERT, GloVe-LSTM, XGBoost) before running the ensemble model.")
    sys.exit(1)

# Load model prediction outputs
df_bert = pd.read_csv("outputs/audience/ensemble_input_DistilBERT_model.csv")
df_glove = pd.read_csv("outputs/audience/ensemble_input_glove_model.csv")
df_xgb = pd.read_csv("outputs/audience/ensemble_input_xgboost_model.csv")

# Create feature matrix from individual model probabilities and confidence scores
X_final = pd.DataFrame({
    "bert_adult": df_bert["proba_adult"],
    "bert_child": df_bert["proba_child"],
    "bert_conf": df_bert["softmax_score"],
    "glove_adult": df_glove["proba_adult"],
    "glove_child": df_glove["proba_child"],
    "glove_conf": df_glove["softmax_score"],
    "xgb_adult": df_xgb["proba_adult"],
    "xgb_child": df_xgb["proba_child"],
    "xgb_conf": df_xgb["softmax_score"]
})
y_final = df_bert["true_label"].values

# Train final Logistic Regression meta-model on the full dataset
final_lr = LogisticRegression(max_iter=1000)
final_lr.fit(X_final, y_final)
final_preds = final_lr.predict(X_final)
final_probs = final_lr.predict_proba(X_final)

# Generate and save confusion matrix
cm = confusion_matrix(y_final, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["adult", "child"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Logistic Regression Ensemble - Confusion Matrix")
plt.tight_layout()
plt.savefig("graphics/ensemble_lr_confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved to: graphics/ensemble_lr_confusion_matrix.png")

# Print performance metrics
print("\n📊 Final Model Performance:")
print(classification_report(y_final, final_preds, target_names=["adult", "child"]))
print("Accuracy:", accuracy_score(y_final, final_preds))
print("Macro F1:", f1_score(y_final, final_preds, average="macro"))
