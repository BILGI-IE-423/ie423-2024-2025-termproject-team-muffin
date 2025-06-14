import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Create output folders
os.makedirs("models/audience", exist_ok=True)
os.makedirs("outputs/audience", exist_ok=True)
os.makedirs("/graphics", exist_ok=True)

# Load data
original_df = pd.read_csv("Training Datasets/Decoded_Dataset_with_Genres_Sentiment_Violence.csv")
augmented_df = pd.read_csv("Training Datasets/train_augmented_child.csv")

original_df = original_df[original_df["audience_group"].isin(["adult", "child"])]
augmented_df = augmented_df[augmented_df["audience_group"] == "child"]

original_df = original_df.dropna(subset=["description", "genres", "sentiment_score", "violence_flag"])
augmented_df = augmented_df.dropna(subset=["description", "genres", "sentiment_score", "violence_flag"])

# Split original data
train_df, temp_df = train_test_split(original_df, test_size=0.3, stratify=original_df["audience_group"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["audience_group"], random_state=42)

# Append augmented child data to train
train_final = pd.concat([train_df, augmented_df], ignore_index=True)

# Tokenizer and sequence length
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_final["description"])
vocab_size = len(tokenizer.word_index) + 1
max_len = 100

def tokenize(df):
    return pad_sequences(tokenizer.texts_to_sequences(df["description"]), maxlen=max_len)

X_train_desc = tokenize(train_final)
X_val_desc = tokenize(val_df)
X_test_desc = tokenize(test_df)

# Genre one-hot encoding
mlb = MultiLabelBinarizer()
train_final["genres_list"] = train_final["genres"].apply(lambda x: x.split(", "))
val_df["genres_list"] = val_df["genres"].apply(lambda x: x.split(", "))
test_df["genres_list"] = test_df["genres"].apply(lambda x: x.split(", "))
mlb.fit(train_final["genres_list"])

X_train_genre = mlb.transform(train_final["genres_list"])
X_val_genre = mlb.transform(val_df["genres_list"])
X_test_genre = mlb.transform(test_df["genres_list"])

# Extra features
X_train_extra = train_final[["sentiment_score", "violence_flag"]].values
X_val_extra = val_df[["sentiment_score", "violence_flag"]].values
X_test_extra = test_df[["sentiment_score", "violence_flag"]].values

# Label encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_final["audience_group"])
y_val = label_encoder.transform(val_df["audience_group"])
y_test = label_encoder.transform(test_df["audience_group"])

y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Load GloVe
embedding_dim = 100
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in embedding_index:
        embedding_matrix[i] = embedding_index[word]

# Class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# Model architecture
input_desc = Input(shape=(max_len,), name="desc_input")
input_genre = Input(shape=(X_train_genre.shape[1],), name="genre_input")
input_extra = Input(shape=(2,), name="extra_input")

x = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(input_desc)
x = SpatialDropout1D(0.3)(x)
x = Bidirectional(LSTM(64))(x)

merged = concatenate([x, input_genre, input_extra])
merged = Dense(64, activation="relu")(merged)
merged = Dropout(0.4)(merged)
output = Dense(2, activation="softmax")(merged)

model = Model(inputs=[input_desc, input_genre, input_extra], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Training
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(
    [X_train_desc, X_train_genre, X_train_extra], y_train_cat,
    validation_data=([X_val_desc, X_val_genre, X_val_extra], y_val_cat),
    epochs=10,
    batch_size=64,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=2
)

# Save model, tokenizer and genre encoder
model.save("models/audience/glove_lstm_model.h5")
with open("models/audience/glove_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("models/audience/genre_mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("✅ Model and encoders saved.")

# Predictions on test set
y_pred_proba = model.predict([X_test_desc, X_test_genre, X_test_extra])
y_pred = np.argmax(y_pred_proba, axis=1)

# Save ensemble CSV
ensemble_df = pd.DataFrame({
    "norm_title": test_df.get("norm_title", test_df.index),
    "description": test_df["description"],
    "true_label": label_encoder.inverse_transform(y_test),
    "predicted_label": label_encoder.inverse_transform(y_pred),
    "proba_adult": y_pred_proba[:, 0],
    "proba_child": y_pred_proba[:, 1],
    "softmax_score": np.max(y_pred_proba, axis=1)
})
ensemble_df.to_csv("outputs/audience/ensemble_input_glove_model.csv", index=False)
print("✅ Ensemble predictions saved.")

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("GloVe + LSTM Ensemble - Confusion Matrix")
plt.tight_layout()
plt.savefig("graphics/glove_lstm_confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved: graphics/glove_lstm_confusion_matrix.png")
