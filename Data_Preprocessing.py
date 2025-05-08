import pandas as pd
import os
import re
import ast
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

# Folder Control
REQUIRED_FOLDERS = ["Raw Datasets", "Training Datasets", "Encoded Datasets"]
for folder in REQUIRED_FOLDERS:
    full_path = os.path.join(os.getcwd(), folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created missing folder: {folder}")


# Normalization Maps
NORMALIZATION_MAPS = {
    "genre": {
        "tv dramas": "drama", "drama movies": "drama", "dramas": "drama", "drama": "drama",
        "tv comedies": "comedy", "comedies": "comedy", "comedy": "comedy",
        "crime tv shows": "crime", "crime dramas": "crime", "crime": "crime",
        "thrillers": "thriller", "thriller": "thriller",
        "documentaries": "documentary", "docuseries": "documentary", "documentary": "documentary",
        "kids' tv": "kids", "children & family movies": "kids", "kids": "kids",
        "lgbtq": "lgbtq", "lgbtq movies": "lgbtq",
        "action & adventure": "action", "tv action & adventure": "action", "tv action and adventure": "action",
        "romantic movies": "romance", "romantic tv shows": "romance",
        "sci-fi & fantasy": "sci-fi", "science fiction": "sci-fi", "tv sci-fi & fantasy": "sci-fi",
        "faith and spirituality": "faith & spirituality", "tv shows": "tv",
        "musicals": "music & musicals", "musical": "music & musicals", "music": "music & musicals", "music videos and concerts": "music & musicals",
        "anime features": "anime", "anime series": "anime",
        "stand-up comedy & talk shows": "stand-up comedy",
        "tv horror": "horror", "horror": "horror", "horror movies": "horror",
        "tv mysteries": "mystery", "mystery": "mystery",
        "sports movies": "sports", "sport": "sports",
        "spanish-language tv shows": "spanish", "spanish language tv shows": "spanish",
        "classic & cult tv": "classic", "classic movies": "classic", "cult movies": "cult",
        "entertainment": "entertainment", "family": "family",
        "historical": "historical", "history": "historical",
        "independent movies": "independent",
        "international": "international", "international movies": "international", "international tv shows": "international", "british tv shows": "international",
        "korean tv shows": "korean", "military and war": "military",
        "movies": "movies", "reality tv": "reality tv",
        "teen tv shows": "teen", "young adult audience": "young adult",
        "suspense": "suspense", "talk show and variety": "talk show",
        "war": "war", "western": "western",
        "tv thrillers": "thriller"
    },
    "age_rating": {
        "TV-Y": "0+", "TV-Y7": "7+", "TV-G": "0+",
        "TV-PG": "10+", "TV-14": "14+", "TV-MA": "18+",
        "G": "0+", "PG": "10+", "PG-13": "13+",
        "R": "16+", "NC-17": "18+"
    },
    "column_names": {
        "title": ["title", "Title"],
        "description": ["description", "Description"],
        "genre": ["genre", "listed_in", "Genre"],
        "rating": ["rating"],
        "release_date": ["release_date"],
        "vote_average": ["vote_average"],
        "vote_count": ["vote_count"],
        "popularity": ["popularity"],
        "genres": ["genres"],
        "duration": ["duration"]
    },
    "allowed_genres": [
        "action", "adventure", "drama", "comedy", "romance",
        "horror", "documentary", "sci-fi", "thriller"
    ]
}

def normalize_title(title):
    if not isinstance(title, str):
        return ""
    cleaned = re.sub(r"\(.*?\)|[^\w\s]", "", title.lower())
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

def normalize_rating_to_group(rating):
    if not isinstance(rating, str):
        return None
    label = NORMALIZATION_MAPS["age_rating"].get(rating.strip().upper())
    if label in ["0+", "7+", "10+"]:
        return "child"
    if label in ["13+", "14+", "16+"]:
        return "teen"
    if label == "18+":
        return "adult"
    return None

def normalize_genres_str(genre_string):
    if not isinstance(genre_string, str):
        return []
    genres = [g.strip().lower() for g in genre_string.split(",")]
    return list({NORMALIZATION_MAPS["genre"].get(g, g) for g in genres})

def get_dataset_paths():
    base = os.getcwd()
    raw_dir = os.path.join(base, "Raw Datasets")
    return {
        "tmdb": os.path.join(raw_dir, "tmdb_raw_data.csv"),
        "netflix": os.path.join(raw_dir, "netflix_titles.csv"),
        "amazon": os.path.join(raw_dir, "amazon_prime_titles.csv"),
        "imdb2": os.path.join(raw_dir, "imdb_movie_dataset.csv")
    }

def normalize_tmdb_row(row):
    title = row.get("title", "")
    release_date = row.get("release_date", "")
    genres_raw = row.get("genres", "")
    return {
        "title": title,
        "norm_title": normalize_title(title),
        "release_year": pd.to_datetime(release_date, errors="coerce").year,
        "vote_average": row.get("vote_average"),
        "vote_count": row.get("vote_count"),
        "popularity": row.get("popularity"),
        "duration": row.get("duration"),
        "normalized_genres": normalize_genres_str(genres_raw)
    }

def normalize_genre_row(row):
    return {
        "norm_title": normalize_title(row.get("title", "")),
        "description": row.get("description", ""),
        "normalized_genres": normalize_genres_str(row.get("genre", ""))
    }

def normalize_age_row(row):
    return {
        "norm_title": normalize_title(row.get("title", "")),
        "description": row.get("description", ""),
        "audience_group": normalize_rating_to_group(row.get("rating", ""))
    }

def load_dataset(name, required_columns):
    paths = get_dataset_paths()
    path = paths.get(name)
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    col_map = {}
    for col in required_columns:
        aliases = NORMALIZATION_MAPS["column_names"].get(col, [col])
        for actual in df.columns:
            if any(alias.lower() in actual.lower() for alias in aliases):
                col_map[actual] = col
                break
    if len(col_map) < len(required_columns):
        return None
    df = df[list(col_map)].rename(columns=col_map)
    df.dropna(subset=required_columns, inplace=True)
    return df

def apply_normalization(df, source_type):
    normalizers = {
        "tmdb": normalize_tmdb_row,
        "genre": normalize_genre_row,
        "age": normalize_age_row
    }
    func = normalizers[source_type]
    return df.apply(func, axis=1, result_type="expand")

def load_and_merge(datasets, required_columns, source_type, join_on_norm_title=None):
    frames = []
    for name in datasets:
        df = load_dataset(name, required_columns)
        if df is not None:
            frames.append(apply_normalization(df, source_type))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="norm_title")
    if join_on_norm_title is not None:
        result = pd.merge(result, join_on_norm_title, on="norm_title", how="inner")
    return result

def save_processed_data(df, columns, filename):
    training_dir = os.path.join(os.getcwd(), "Training Datasets")
    full_path = os.path.join(training_dir, filename)
    df.to_csv(full_path, columns=columns, index=False)
    print(f"Saved to {full_path}")

# Clean genre
def clean_and_filter_genres(x):
    allowed_genres = NORMALIZATION_MAPS["allowed_genres"]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        genres = x
    elif isinstance(x, str):
        try:
            genres = ast.literal_eval(x)
            if not isinstance(genres, list):
                genres = [genres]
        except Exception:
            genres = [x]
    else:
        genres = []
    return [g.lower() for g in genres if isinstance(g, str) and g.lower() in allowed_genres]

def categorize_duration(mins):
    if pd.isna(mins):
        return "unknown"
    if mins < 90:
        return "short"
    elif mins <= 120:
        return "medium"
    else:
        return "long"

# Generate and save datasets
df_age = load_and_merge(["netflix", "amazon"], ["title", "description", "rating"], "age")
save_processed_data(df_age, ["norm_title", "description", "audience_group"], "audience_group_training_data.csv")

df_genre = load_and_merge(["netflix", "amazon", "imdb2"], ["title", "description", "genre"], "genre")
save_processed_data(df_genre, ["norm_title", "description", "normalized_genres"], "genre_training_data.csv")

df_duration = load_and_merge(["tmdb"],
    ["title", "release_date", "vote_average", "vote_count", "popularity", "genres", "duration"],
    "tmdb", join_on_norm_title=df_age[["norm_title", "audience_group"]]
)
save_processed_data(df_duration, ["title", "norm_title", "release_year", "vote_average", "vote_count", "popularity", "duration", "normalized_genres", "audience_group"],
                    "duration_popularity_training_data.csv")

# One-hot encode everything
df_duration["normalized_genres"] = df_duration["normalized_genres"].apply(clean_and_filter_genres)
df_duration = df_duration[df_duration["normalized_genres"].map(lambda x: len(x) > 0)]
df_duration["duration_group"] = df_duration["duration"].apply(categorize_duration)

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df_duration["normalized_genres"]), columns=mlb.classes_)
df_duration = df_duration.reset_index(drop=True).join(genre_encoded)
df_duration = pd.get_dummies(df_duration, columns=["audience_group", "duration_group"])

# Encoded Outputs Directory
encoded_dir = os.path.join(os.getcwd(), "Encoded Datasets")
df_duration.to_csv(os.path.join(encoded_dir, "duration_training_encoded.csv"), index=False)

# Audience_group one-hot
df_age_encoded = df_age.copy()
df_age_encoded = pd.get_dummies(df_age_encoded, columns=["audience_group"])
df_age_encoded.to_csv(os.path.join(encoded_dir, "audience_group_training_encoded.csv"), index=False)

# Normalized_genres -> one-hot
df_genre_encoded = df_genre.copy()
df_genre_encoded["normalized_genres"] = df_genre_encoded["normalized_genres"].apply(clean_and_filter_genres)
df_genre_encoded = df_genre_encoded[df_genre_encoded["normalized_genres"].map(lambda x: len(x) > 0)]

mlb_genre = MultiLabelBinarizer()
genre_onehot = pd.DataFrame(mlb_genre.fit_transform(df_genre_encoded["normalized_genres"]), columns=mlb_genre.classes_)
df_genre_encoded = df_genre_encoded.reset_index(drop=True).join(genre_onehot)
df_genre_encoded.to_csv(os.path.join(encoded_dir, "genre_training_encoded.csv"), index=False)
