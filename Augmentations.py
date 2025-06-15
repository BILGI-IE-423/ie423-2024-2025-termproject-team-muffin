import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm

# ================================
# Load Dataset
# ================================
df = pd.read_csv("Training Datasets/multiinput_amazon_netflix_genre.csv")

# Parse genre column if stored as string
df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# ================================
# 1. Genre-Based Augmentation
# ================================
genre_targets = ['sci-fi', 'thriller', 'horror']
genre_augmenter = naw.SynonymAug(aug_src='wordnet')
genre_augmented_rows = []

for genre in genre_targets:
    genre_rows = df[df['genres'].apply(lambda g: genre in g)]
    for _, row in tqdm(genre_rows.iterrows(), total=len(genre_rows), desc=f"Augmenting genre: {genre}"):
        new_row = row.copy()
        try:
            new_row['description'] = genre_augmenter.augment(row['description'])
            genre_augmented_rows.append(new_row)
        except:
            continue

# Save genre-augmented data
df_genre_augmented = pd.DataFrame(genre_augmented_rows)
df_genre_augmented.to_csv("augmented_genre_only.csv", index=False)

# ================================
# 2. Child Audience-Based Augmentation
# ================================
df_child = df[df["audience_group"] == "child"].copy()
child_augmenter = naw.SynonymAug(aug_src='wordnet')

augmented_child_descriptions = []
for text in tqdm(df_child["description"], desc="Augmenting child descriptions"):
    try:
        augmented_text = child_augmenter.augment(text)
        augmented_child_descriptions.append(augmented_text)
    except:
        continue

df_augmented_child = df_child.iloc[:len(augmented_child_descriptions)].copy()
df_augmented_child["description"] = augmented_child_descriptions
df_augmented_child.to_csv("Training Datasets/audience/train_augmented_child.csv", index=False)
