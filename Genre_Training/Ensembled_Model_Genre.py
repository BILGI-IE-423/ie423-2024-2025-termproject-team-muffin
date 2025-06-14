import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Read ensemble inputs
df_bert = pd.read_csv("outputs/genre/ensemble_input_distilbert_model.csv")
df_lstm = pd.read_csv("outputs/genre/ensemble_input_glove_model.csv")
df_xgb = pd.read_csv("outputs/genre/ensemble_input_xgboost_model.csv")

# 2. Merge based on common test set
ensemble_df = df_bert[["norm_title"]].copy()
ensemble_df = ensemble_df.merge(df_lstm, on="norm_title", suffixes=("", "_lstm"))
ensemble_df = ensemble_df.merge(df_xgb, on="norm_title", suffixes=("", "_xgb"))

# 3. Genre classes
genre_labels = [col.replace("proba_", "") for col in df_bert.columns if col.startswith("proba_")]

# 4. Feature matrix (X)
X = pd.DataFrame()
for g in genre_labels:
    X[f"bert_{g}"] = df_bert[f"proba_{g}"]
    X[f"lstm_{g}"] = df_lstm[f"proba_{g}"]
    X[f"xgb_{g}"] = df_xgb[f"proba_{g}"]

# 5. Reconstruct true labels
# Since each model shares the same test order, one is sufficient
df_true = pd.read_csv("Training Datasets/Decoded_Dataset_with_Genres_Sentiment_Violence.csv")
df_true["genres"] = df_true["genres"].apply(eval)
test_titles = df_bert["norm_title"].tolist()

genre_true_map = dict(df_true[["norm_title", "genres"]].values)
true_genres_list = [genre_true_map.get(t, []) for t in test_titles]

mlb = MultiLabelBinarizer(classes=genre_labels)
y_true = mlb.fit_transform(true_genres_list)

# 6. Train ensemble model
ensemble_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
ensemble_model.fit(X, y_true)

y_pred = ensemble_model.predict(X)

# 7. Performance output
print("\n📊 Ensemble Model Performance:")
print(classification_report(y_true, y_pred, target_names=genre_labels, zero_division=0))
