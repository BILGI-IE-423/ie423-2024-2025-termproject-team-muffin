import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create required folders
os.makedirs("models/audience", exist_ok=True)
os.makedirs("outputs/audience", exist_ok=True)

# -----------------------------
# 1. Load datasets
# -----------------------------
original_df = pd.read_csv("Training Datasets/Decoded_Dataset_with_Genres_Sentiment_Violence.csv")
augmented_df = pd.read_csv("Training Datasets/train_augmented_child.csv")

# Filter samples
original_df = original_df[original_df["audience_group"].isin(["adult", "child"])]
augmented_df = augmented_df[augmented_df["audience_group"] == "child"]

original_df = original_df.dropna(subset=["description", "genres", "sentiment_score", "violence_flag"])
augmented_df = augmented_df.dropna(subset=["description", "genres", "sentiment_score", "violence_flag"])

# -----------------------------
# 2. Split original data (no augment in val/test)
# -----------------------------
train_df, temp_df = train_test_split(
    original_df, test_size=0.3, stratify=original_df["audience_group"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["audience_group"], random_state=42
)

# -----------------------------
# 3. Merge train + augmented only in train
# -----------------------------
train_final = pd.concat([train_df, augmented_df], ignore_index=True)

# -----------------------------
# 4. Encode labels
# -----------------------------
label_encoder = LabelEncoder()
train_final["label"] = label_encoder.fit_transform(train_final["audience_group"])
val_df["label"] = label_encoder.transform(val_df["audience_group"])
test_df["label"] = label_encoder.transform(test_df["audience_group"])

# -----------------------------
# 5. TF-IDF vectorization
# -----------------------------
tfidf_desc = TfidfVectorizer(max_features=1000)
tfidf_genre = TfidfVectorizer(max_features=50)

X_train_desc = tfidf_desc.fit_transform(train_final["description"]).toarray()
X_val_desc = tfidf_desc.transform(val_df["description"]).toarray()
X_test_desc = tfidf_desc.transform(test_df["description"]).toarray()

X_train_genre = tfidf_genre.fit_transform(train_final["genres"]).toarray()
X_val_genre = tfidf_genre.transform(val_df["genres"]).toarray()
X_test_genre = tfidf_genre.transform(test_df["genres"]).toarray()

# -----------------------------
# 6. Numeric features
# -----------------------------
X_train_meta = train_final[["sentiment_score", "violence_flag"]].values
X_val_meta = val_df[["sentiment_score", "violence_flag"]].values
X_test_meta = test_df[["sentiment_score", "violence_flag"]].values

# -----------------------------
# 7. Combine all features
# -----------------------------
X_train = np.hstack([X_train_desc, X_train_genre, X_train_meta])
X_val = np.hstack([X_val_desc, X_val_genre, X_val_meta])
X_test = np.hstack([X_test_desc, X_test_genre, X_test_meta])

y_train = train_final["label"].values
y_val = val_df["label"].values
y_test = test_df["label"].values

# -----------------------------
# 8. Train XGBoost
# -----------------------------
model = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# -----------------------------
# 9. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -----------------------------
# 10. Save model and transformers
# -----------------------------
joblib.dump(model, "models/audience/xgboost_model.joblib")
pickle.dump(tfidf_desc, open("models/audience/tfidf_desc.pkl", "wb"))
pickle.dump(tfidf_genre, open("models/audience/tfidf_genre.pkl", "wb"))
pickle.dump(label_encoder, open("models/audience/label_encoder.pkl", "wb"))

# -----------------------------
# 11. Save ensemble output
# -----------------------------
ensemble_df = pd.DataFrame({
    "norm_title": test_df.get("norm_title", test_df.index),
    "description": test_df["description"],
    "true_label": label_encoder.inverse_transform(y_test),
    "predicted_label": label_encoder.inverse_transform(y_pred),
    "proba_class_0": y_pred_proba[:, 0],
    "proba_class_1": y_pred_proba[:, 1],
    "softmax_score": np.max(y_pred_proba, axis=1)
})

class_names = list(label_encoder.classes_)
ensemble_df.rename(columns={
    "proba_class_0": f"proba_{class_names[0]}",
    "proba_class_1": f"proba_{class_names[1]}"
}, inplace=True)

ensemble_df.to_csv("outputs/audience/ensemble_input_xgboost_model.csv", index=False)

print("✅ Model and outputs saved.")

# -----------------------------
# 12. Save confusion matrix
# -----------------------------
os.makedirs("graphics", exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("XGBoost Ensemble - Confusion Matrix")
plt.tight_layout()
plt.savefig("graphics/xgboost_confusion_matrix.png")
plt.close()

print("✅ Confusion matrix saved: graphics/xgboost_confusion_matrix.png")

# -----------------------------
# 13. Save classification report to CSV
# -----------------------------
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("outputs/audience/classification_report.csv", index=True)

print("✅ Classification report saved: outputs/audience/classification_report.csv")
