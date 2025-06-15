import pandas as pd
from textblob import TextBlob
import re

# Load your dataset
df = pd.read_csv("Training Datasets/multiinput_amazon_netflix_genre.csv")  # Replace with your actual path

# 1. Sentiment Score Calculation using TextBlob
def compute_sentiment(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return 0.0
    return TextBlob(text).sentiment.polarity

df["sentiment_score"] = df["description"].apply(compute_sentiment)

# 2. Violence Flag Detection (keyword-based)
violence_keywords = [
    "kill", "murder", "blood", "war", "weapon", "gun", "fight", "torture", "assault",
    "explosion", "hostage", "terror", "abuse", "rape", "shoot", "knife", "crime", "violence"
]

def detect_violence(text):
    if pd.isna(text):
        return 0
    text_lower = text.lower()
    return int(any(re.search(rf"\b{kw}\b", text_lower) for kw in violence_keywords))

df["violence_flag"] = df["description"].apply(detect_violence)

# Save to new CSV
df.to_csv("Training Datasets/Decoded_Dataset_with_Genres_Sentiment_Violence", index=False)
