import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm

# Load your main dataset (or kullanıyorsan df'yi doğrudan bırakabilirsin)
df = pd.read_csv("your_main_dataset.csv")  # bu satırı sil, df zaten varsa

# Parse genre string to list (sadece gerekiyorsa)
df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Genres to augment
target_genres = ['sci-fi', 'thriller', 'horror']

# Prepare augmenter
augmenter = naw.SynonymAug(aug_src='wordnet')

# Store augmented rows
augmented_rows = []

for genre in target_genres:
    rows = df[df['genres'].apply(lambda g: genre in g)]
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"Augmenting {genre}"):
        new_row = row.copy()
        try:
            new_row['description'] = augmenter.augment(row['description'])
            augmented_rows.append(new_row)
        except:
            continue  # problematic text, skip

# Convert to DataFrame and save
df_augmented = pd.DataFrame(augmented_rows)
df_augmented.to_csv("augmented_genre_only.csv", index=False)



import pandas as pd
import nlpaug.augmenter.word as naw

# Load original data
df = pd.read_csv("Training Datasets/audience/audience_group_training_data.csv")

# Filter only 'child' labeled data
df_child = df[df["audience_group"] == "child"].copy()

# Initialize augmenter
augmenter = naw.SynonymAug(aug_src='wordnet')  # veya başka bir augmenter

# Apply augmentation
augmented_texts = []
for text in df_child["description"]:
    try:
        aug_text = augmenter.augment(text)
        augmented_texts.append(aug_text)
    except:
        continue

# Create DataFrame
df_aug = df_child.iloc[:len(augmented_texts)].copy()
df_aug["description"] = augmented_texts

# Save to CSV
df_aug.to_csv("Training Datasets/audience/train_augmented_child.csv", index=False)
