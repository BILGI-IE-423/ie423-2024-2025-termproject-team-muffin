import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Create necessary folders
os.makedirs("models/audience", exist_ok=True)
os.makedirs("outputs/audience", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
original_df = pd.read_csv("Training Datasets/Decoded_Dataset_with_Genres_Sentiment_Violence.csv")
augmented_df = pd.read_csv("Training Datasets/train_augmented_child.csv")

# Filter only adult and child samples
original_df = original_df[original_df["audience_group"].isin(["adult", "child"])].copy()
augmented_df = augmented_df[augmented_df["audience_group"] == "child"].copy()

# Select relevant columns
original_df = original_df[['norm_title', 'description', 'audience_group']].dropna()
augmented_df = augmented_df[['norm_title', 'description', 'audience_group']].dropna()

# Encode labels
label2idx = {'adult': 0, 'child': 1}
original_df['label'] = original_df['audience_group'].map(label2idx)
augmented_df['label'] = augmented_df['audience_group'].map(label2idx)

# Split train/val/test
texts = original_df['description'].values
labels = original_df['label'].values
indices = original_df.index.values

X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    texts, labels, indices, test_size=0.15, stratify=labels, random_state=42
)
val_ratio = 0.15 / (1 - 0.15)
X_train_orig, X_val, y_train_orig, y_val, idx_train_orig, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=val_ratio, stratify=y_temp, random_state=42
)

# Append augmented child data to training set
X_train = np.concatenate([X_train_orig, augmented_df['description'].values])
y_train = np.concatenate([y_train_orig, augmented_df['label'].values])
idx_train = np.concatenate([idx_train_orig, augmented_df.index.values])

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
max_len = 128

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, indices):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_len)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = indices
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        item['idx'] = self.indices[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(X_train, y_train, idx_train)
val_dataset = TextDataset(X_val, y_val, idx_val)
test_dataset = TextDataset(X_test, y_test, idx_test)

# Weighted sampler
class_sample_count = np.bincount(y_train)
weights = 1. / class_sample_count
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model class
class MovieClassifier(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(768, 256)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(256, 2)
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        x = self.dropout(self.pre_classifier(output))
        return self.classifier(x)

model = MovieClassifier(dropout_p=0.44848889551297877).to(device)
optimizer = AdamW(model.parameters(), lr=3.7919251898688514e-05)
criterion = nn.CrossEntropyLoss()

# Training and evaluation functions
def train_epoch(model, loader):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, true, probs, indices = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            idxs = batch['idx']
            logits = model(input_ids, attention_mask)
            softmaxed = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = np.argmax(softmaxed, axis=1)
            preds.extend(pred_classes)
            probs.extend(softmaxed)
            true.extend(labels)
            indices.extend(idxs)
    return preds, true, probs, indices

# Training loop with early stopping
best_f1 = 0
patience = 3
patience_counter = 0
best_model_state = None

for epoch in range(20):
    train_epoch(model, train_loader)
    val_preds, val_true, _, _ = evaluate(model, val_loader)
    val_f1 = f1_score(val_true, val_preds, average='macro')
    print(f"Epoch {epoch+1}, Validation Macro F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# Load best model
model.load_state_dict(best_model_state)

# Test the model
test_preds, test_true, test_probs, test_indices = evaluate(model, test_loader)

print("\nTest Set Results:")
print(classification_report(test_true, test_preds, target_names=['adult', 'child']))
print(f"Accuracy: {accuracy_score(test_true, test_preds):.4f}")
print(f"Macro F1: {f1_score(test_true, test_preds, average='macro'):.4f}")

# Save model
torch.save(model.state_dict(), "models/audience/model_adult_child_only.pth")
print("✅ Model saved to: models/audience/model_adult_child_only.pth")

# Export ensemble input CSV
output_df = pd.concat([
    original_df.loc[idx_test].reset_index(drop=True),
    pd.DataFrame({
        'predicted_label': test_preds,
        'proba_adult': [p[0] for p in test_probs],
        'proba_child': [p[1] for p in test_probs],
        'softmax_score': np.max(test_probs, axis=1)
    })
], axis=1)

output_df = output_df[['norm_title', 'description', 'label', 'predicted_label', 'proba_adult', 'proba_child', 'softmax_score']]
output_df.rename(columns={'label': 'true_label'}, inplace=True)

output_df.to_csv("outputs/audience/ensemble_DistilBERT_model", index=False)
print("✅ Ensemble output saved: outputs/audience/ensemble_DistilBERT_model.csv")


