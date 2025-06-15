import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your dataset
df = pd.read_csv('Training Datasets/duration_popularity_training_data.csv')
df = df.dropna()

# 2. Create binary indicator columns for duration categories
df['duration_short']  = (df['duration'] < 80).astype(int)
df['duration_medium'] = ((df['duration'] >= 80) & (df['duration'] <= 120)).astype(int)
df['duration_long']   = (df['duration'] > 120).astype(int)

# 3. Create binary indicator columns for audience groups
df['audience_adult'] = (df['audience_group'] == 'adult').astype(int)
df['audience_child'] = (df['audience_group'] == 'child').astype(int)
df['audience_teen']  = (df['audience_group'] == 'teen').astype(int)

# 4. Create a binary popularity label
threshold = 7  # adjust as needed
df['popularity_label'] = df['popularity'].apply(
    lambda x: 'popular' if x >= threshold else 'not_popular'
)

# 5. Derive single categorical labels
df['audience_label'] = (
    df[['audience_adult', 'audience_child', 'audience_teen']]
    .idxmax(axis=1)
    .str.replace('audience_', '', regex=False)
)
df['duration_label'] = (
    df[['duration_short', 'duration_medium', 'duration_long']]
    .idxmax(axis=1)
    .str.replace('duration_', '', regex=False)
)

# 6. Plot popularity distribution by duration for each audience group
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for i, aud in enumerate(['adult', 'teen', 'child']):
    sub = df[df['audience_label'] == aud]
    if not sub.empty:
        ct = pd.crosstab(
            sub['duration_label'],
            sub['popularity_label'],
            normalize='index'
        ) * 100
        print(f"\n======= Audience Group: {aud.capitalize()} =======")
        print(ct.round(1))

        pct = ct.reset_index()
        pct.plot(
            x='duration_label',
            y=['popular', 'not_popular'],
            kind='bar',
            ax=axes[i],
            legend=(aud == 'teen'),
            title=f"{aud.capitalize()} Audience",
            ylabel="Percentage (%)"
        )
        axes[i].set_xlabel("Duration Category")
    else:
        print(f"\n=== Audience Group: {aud.capitalize()} ===")
        print("No data for this group.")
        axes[i].set_title(f"{aud.capitalize()} Audience (No Data)")
        axes[i].set_xlabel("Duration Category")

plt.tight_layout()
plt.show()

# 7. Normalize popularity and compute correlations
pop_min, pop_max = df['popularity'].min(), df['popularity'].max()
df['pop_norm'] = (df['popularity'] - pop_min) / (pop_max - pop_min)

# Overall correlations
pearson_overall = df['duration'].corr(df['popularity'], method='pearson')
spearman_overall = df['duration'].corr(df['popularity'], method='spearman')
print(f"\nOverall Pearson correlation:  {pearson_overall:.3f}")
print(f"Overall Spearman correlation: {spearman_overall:.3f}\n")

# Per-audience correlations
for aud in ['adult', 'child', 'teen']:
    sub = df[df['audience_label'] == aud]
    if len(sub) > 1:
        p = sub['duration'].corr(sub['popularity'], method='pearson')
        s = sub['duration'].corr(sub['popularity'], method='spearman')
        print(f"{aud.capitalize()} audience:")
        print(f"  Pearson:  {p:.3f}")
        print(f"  Spearman: {s:.3f}\n")
    else:
        print(f"{aud.capitalize()} audience: insufficient data to compute correlation.\n")
