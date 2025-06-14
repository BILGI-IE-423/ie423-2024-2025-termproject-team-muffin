import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW

# ========= 1. Load Data =========
train_df = pd.read_csv("Training Datasets/genre/train_iterative.csv")
val_df = pd.read_csv("Training Datasets/genre/val_iterative.csv")
test_df = pd.read_csv("Training Datasets/genre/test_iterative.csv")

train_df["genres"] = train_df["genres"].apply(eval)
val_df["genres"] = val_df["genres"].apply(eval)
test_df["genres"] = test_df["genres"].apply(eval)

genre_columns = ['action', 'sci-fi', 'comedy', 'documentary', 'drama', 'horror', 'thriller', 'romance']

def binarize(df):
    for genre in genre_columns:
        df[genre] = df["genres"].apply(lambda x: int(genre in [g.lower() for g in x]))
    return df

train_df = binarize(train_df)
val_df = binarize(val_df)
test_df = binarize(test_df)

X_train, y_train = train_df["description"].astype(str).values, train_df[genre_columns].values.astype(np.float32)
X_val, y_val = val_df["description"].astype(str).values, val_df[genre_columns].values.astype(np.float32)
X_test, y_test = test_df["description"].astype(str).values, test_df[genre_columns].values.astype(np.float32)

# ========= 2. Tokenizer and Dataset =========
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
max_len = 128

class GenreDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = GenreDataset(X_train, y_train)
val_dataset = GenreDataset(X_val, y_val)
test_dataset = GenreDataset(X_test, y_test)

# Weighted sampling
y_train_sum = y_train.sum(axis=0)
weights = 1. / (y_train_sum + 1e-6)
sample_weights = [weights[np.where(row == 1)[0]].mean() if np.any(row == 1) else 1e-6 for row in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========= 3. Model =========
class GenreClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(0.2)
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        x = self.dropout(self.pre_classifier(x))
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GenreClassifier(num_labels=len(genre_columns)).to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)
criterion = nn.BCEWithLogitsLoss()

# ========= 4. Training =========
def evaluate(model, loader):
    model.eval()
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, attention_mask).cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
            all_probs.append(probs)
            all_true.append(labels)
    return np.vstack(all_preds), np.vstack(all_true), np.vstack(all_probs)

best_f1, best_state = 0, None
patience, patience_counter = 3, 0

for epoch in range(20):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    val_preds, val_true, _ = evaluate(model, val_loader)
    val_f1 = f1_score(val_true, val_preds, average="macro")
    print(f"Epoch {epoch+1} | Val Macro F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break

# ========= 5. Save and Output =========
model.load_state_dict(best_state)
test_preds, test_true, test_probs = evaluate(model, test_loader)

os.makedirs("outputs/genre", exist_ok=True)
os.makedirs("models/genre", exist_ok=True)

# Save classification report
report = classification_report(test_true, test_preds, target_names=genre_columns, zero_division=0)
with open("outputs/genre/classification_report_distilbert.txt", "w") as f:
    f.write(report)

# Save ensemble input CSV
df_ensemble = pd.DataFrame({f"proba_{g}": test_probs[:, i] for i, g in enumerate(genre_columns)})
df_ensemble["softmax_score"] = np.mean(test_probs, axis=1)
df_ensemble["norm_title"] = test_df.reset_index(drop=True)["norm_title"]
cols = ["norm_title"] + [f"proba_{g}" for g in genre_columns] + ["softmax_score"]
df_ensemble = df_ensemble[cols]
df_ensemble.to_csv("outputs/genre/ensemble_input_distilbert_model.csv", index=False)

# Save model
torch.save(model.state_dict(), "models/genre/distilbert_genre_model.pth")
print("✅ Model and output saved in BERT-compatible format.")
