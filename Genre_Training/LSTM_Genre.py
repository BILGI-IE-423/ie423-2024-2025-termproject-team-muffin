import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Concatenate, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# ========== 1. Create Required Folders ==========
os.makedirs("graphics/genre", exist_ok=True)
os.makedirs("models/genre", exist_ok=True)
os.makedirs("outputs/genre", exist_ok=True)

# ========== 2. Load Files ==========
train_df = pd.read_csv("Training Datasets/genre/train_iterative.csv")
val_df = pd.read_csv("Training Datasets/genre/val_iterative.csv")
test_df = pd.read_csv("Training Datasets/genre/test_iterative.csv")

train_df["genres"] = train_df["genres"].apply(eval)
val_df["genres"] = val_df["genres"].apply(eval)
test_df["genres"] = test_df["genres"].apply(eval)

# ========== 3. Sorting ==========
train_df = train_df.sort_values("norm_title").reset_index(drop=True)
val_df = val_df.sort_values("norm_title").reset_index(drop=True)
test_df = test_df.sort_values("norm_title").reset_index(drop=True)

# ========== 4. Prepare Inputs ==========
def prepare_inputs(df, tokenizer=None, fit=False):
    text = (df["norm_title"] + " " + df["description"]).astype(str).values
    if fit:
        tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
        tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    X_text = pad_sequences(sequences, maxlen=200, padding="post")
    X_extra = np.vstack([df["sentiment_score"], df["violence_flag"]]).T
    return X_text, X_extra, tokenizer

X_train_text, X_train_extra, tokenizer = prepare_inputs(train_df, fit=True)
X_val_text, X_val_extra, _ = prepare_inputs(val_df, tokenizer)
X_test_text, X_test_extra, _ = prepare_inputs(test_df, tokenizer)

# ========== 5. Prepare Labels ==========
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df["genres"])
y_val = mlb.transform(val_df["genres"])
y_test = mlb.transform(test_df["genres"])
genre_labels = mlb.classes_

# ========== 6. Load GloVe Embeddings ==========
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    if i < 30000 and word in embedding_index:
        embedding_matrix[i] = embedding_index[word]

# ========== 7. Attention Layer ==========
class AttentionLayer(Layer):
    def call(self, inputs):
        score = tf.nn.tanh(inputs)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

# ========== 8. Define Model ==========
input_text = Input(shape=(200,))
input_extra = Input(shape=(2,))

embedding = Embedding(
    input_dim=len(tokenizer.word_index) + 1,
    output_dim=100,
    weights=[embedding_matrix],
    input_length=200,
    trainable=False)(input_text)

x = Bidirectional(LSTM(64, return_sequences=True))(embedding)
x = AttentionLayer()(x)
x = Concatenate()([x, input_extra])
x = Dropout(0.5)(x)
output = Dense(len(genre_labels), activation="sigmoid")(x)

model = Model(inputs=[input_text, input_extra], outputs=output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# ========== 9. Callbacks ==========
checkpoint = ModelCheckpoint("models/genre/glove_bilstm_attention_model.h5", save_best_only=True, monitor="val_loss")
logger = CSVLogger("models/genre/glove_bilstm_attention_log.csv")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ========== 10. Train Model ==========
history = model.fit(
    [X_train_text, X_train_extra], y_train,
    validation_data=([X_val_text, X_val_extra], y_val),
    epochs=20,
    batch_size=64,
    callbacks=[checkpoint, logger, early_stop]
)

# ========== 11. Plot Training History ==========
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("GloVe + BiLSTM + Attention Genre Prediction")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("graphics/genre/glove_bilstm_attention_loss.png")
plt.close()

# ========== 12. Prediction & Save Ensemble Inputs ==========
pred_proba = model.predict([X_test_text, X_test_extra])
softmax_score = np.mean(pred_proba, axis=1)
df_pred = pd.DataFrame(pred_proba, columns=[f"proba_{label}" for label in genre_labels])
df_pred["softmax_score"] = softmax_score
df_pred["norm_title"] = test_df["norm_title"].values
df_pred = df_pred[["norm_title"] + [f"proba_{label}" for label in genre_labels] + ["softmax_score"]]
df_pred.to_csv("outputs/genre/ensemble_input_glove_model.csv", index=False)

# ========== 13. Classification Report ==========
y_pred_bin = (pred_proba >= 0.5).astype(int)
report = classification_report(y_test, y_pred_bin, target_names=genre_labels, zero_division=0)
print(report)

with open("models/genre/glove_bilstm_classification_report.txt", "w") as f:
    f.write(report)
