import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import pickle

# ========== 1. Create Folders ==========
os.makedirs("outputs/genre", exist_ok=True)
os.makedirs("models/genre", exist_ok=True)

# ========== 2. Load Data ==========
train_df = pd.read_csv("Training Datasets/genre/train_iterative.csv")
val_df = pd.read_csv("Training Datasets/genre/val_iterative.csv")
test_df = pd.read_csv("Training Datasets/genre/test_iterative.csv")

train_df["genres"] = train_df["genres"].apply(eval)
val_df["genres"] = val_df["genres"].apply(eval)
test_df["genres"] = test_df["genres"].apply(eval)

# ========== 3. Sort by Normalized Title ==========
train_df = train_df.sort_values("norm_title").reset_index(drop=True)
val_df = val_df.sort_values("norm_title").reset_index(drop=True)
test_df = test_df.sort_values("norm_title").reset_index(drop=True)

# ========== 4. Prepare TF-IDF + Extra Features ==========
def get_text(df):
    return (df["norm_title"] + " " + df["description"]).astype(str).values

tfidf = TfidfVectorizer(max_features=30000)
X_train_text = tfidf.fit_transform(get_text(train_df))
X_val_text = tfidf.transform(get_text(val_df))
X_test_text = tfidf.transform(get_text(test_df))

X_train_extra = train_df[["sentiment_score", "violence_flag"]].values
X_val_extra = val_df[["sentiment_score", "violence_flag"]].values
X_test_extra = test_df[["sentiment_score", "violence_flag"]].values

X_train = hstack([X_train_text, X_train_extra])
X_val = hstack([X_val_text, X_val_extra])
X_test = hstack([X_test_text, X_test_extra])

# ========== 5. Prepare Labels ==========
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df["genres"])
y_val = mlb.transform(val_df["genres"])
y_test = mlb.transform(test_df["genres"])
genre_labels = mlb.classes_

# ========== 6. Train Model ==========
xgb = XGBClassifier(tree_method="hist", eval_metric="logloss", use_label_encoder=False)
model = MultiOutputClassifier(xgb)
model.fit(X_train, y_train)

# ========== 7. Evaluation ==========
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=genre_labels, zero_division=0)
print(report)

with open("models/genre/classification_report_xgboost.txt", "w") as f:
    f.write(report)

# ========== 8. Output for Ensemble ==========
y_pred_proba = model.predict_proba(X_test)
proba_matrix = np.column_stack([prob[:, 1] for prob in y_pred_proba])

df_proba = pd.DataFrame(proba_matrix, columns=[f"proba_{label}" for label in genre_labels])
df_proba["softmax_score"] = np.mean(proba_matrix, axis=1)
df_proba["norm_title"] = test_df["norm_title"].values
df_proba = df_proba[["norm_title"] + [f"proba_{label}" for label in genre_labels] + ["softmax_score"]]
df_proba.to_csv("outputs/genre/ensemble_input_xgboost_model.csv", index=False)

# ========== 9. Save Model ==========
with open("models/genre/xgboost_genre_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/genre/xgboost_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/genre/xgboost_genre_mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
