# Predicting Missing Metadata Fields in Streaming Content Using Text-Based Models

##  Abstract
This project aims to predicting missing metadata fields—such as genre and target age group in streaming content datasets using Natural Language Processing and machine learning techniques. When one of the key elements (genre or age group) is missing the remaining available fields will be used to predict the missing information. Additionally, the relationship between content duration and popularity will be analyzed to understand audience preferences. The project focuses entirely on text-based features and uses publicly available streaming content datasets.

##  Scope of the Project


The scope of this project is to develop and apply text-based machine learning solutions for enhancing the quality of streaming content datasets, with a particular focus on addressing gaps in metadata. By using textual information such as content descriptions the project will build predictive models to estimate missing values for crucial fields like genre and age group. Furthermore it will conduct a statistical exploration of how the duration of content correlates with its popularity among different audiences.

##  Research Questions
1. How does the duration of streaming content (short vs. long) affect popularity among different audience groups?
2. How accurately can the genre of streaming content be predicted from its description?  
3. How accurately can the target age group be predicted from its description and genre?  



#  About Preprocessing Data

##  Datasets to Be Used

The project utilizes data from four major sources, each serving a distinct purpose in the training pipeline:

| **Dataset**       | **Filename**                 | **Purpose**                                      |
|-------------------|------------------------------|--------------------------------------------------|
| Netflix Titles    | `netflix_titles.csv`         | Source of descriptions, genres, and age ratings |
| Amazon Prime      | `amazon_prime_titles.csv`    | Complementary source for genre and age data     |
| IMDb Movie Data   | `imdb_movie_dataset.csv`     | Additional descriptions and genre diversity     |
| TMDb Metadata     | `tmdb_raw_data.csv`          | Contains popularity metrics, duration, and votes. Fetched using TMDb API from [themoviedb.org](https://www.themoviedb.org/)|

Each dataset is selectively normalized and merged according to its structure. While Netflix and Amazon are used mainly for classification tasks (genre and audience), TMDb is used for analyzing numeric trends like content popularity and duration. IMDb provides additional description and genre information to support the genre classification model.

This section explains how the raw data from multiple streaming content platforms was processed and transformed into machine learning-ready formats for genre classification, audience group prediction, and duration-popularity analysis.

---

## 🔹 Step 1: Define Normalization Maps

A dictionary named `NORMALIZATION_MAPS` was created to ensure consistency across sources:

- **Genre mapping**: Converts vague or platform-specific genres (e.g., "tv dramas") to standardized ones like `"drama"`.
- **Age rating mapping**: Groups ratings (e.g., "TV-14") into `"child"`, `"teen"`, or `"adult"`.
- **Column aliasing**: Supports flexible matching of column names.
- **Allowed genres**: Restricts training data to selected target genres.

---

## 🔹 Step 2: Create Utility Functions

Several utility functions were implemented to standardize the values of key metadata fields across diverse datasets:

- `normalize_title(title)`: Cleans a title string by converting it to lowercase, removing punctuation and articles like "the", "a", or "an", and collapsing whitespace. This ensures that title-based joins between datasets are reliable.
- `normalize_rating_to_group(rating)`: Maps specific age ratings (e.g., "PG-13", "TV-MA") into broader audience categories: "child", "teen", or "adult".
- `normalize_genres_str(genre_string)`: Splits and cleans genre strings, maps ambiguous or compound genres to standardized forms (e.g., "tv comedies" → "comedy").
- `categorize_duration(mins)`: Categorizes numeric durations into labels: "short" (<80 mins), "medium" (80–120 mins), or "long" (>120 mins).

These modular functions were applied later during row-level normalization to ensure all content was consistently and meaningfully represented.

---

## 🔹 Step 3: Normalize Rows from Different Sources

Each source dataset uses different naming conventions and metadata formats. To handle this, we applied row-level normalization functions specific to each type of dataset:

- `normalize_tmdb_row()`: Extracts and transforms TMDb fields such as `vote_average`, `popularity`, and `duration`, while also standardizing genres and title.
- `normalize_genre_row()`: Processes datasets (e.g., Netflix, Amazon, IMDb) by extracting the description and converting genre strings into a cleaned list.
- `normalize_age_row()`: Retrieves content descriptions and maps raw rating labels into unified age categories.

Each row in a dataset is transformed via the appropriate function using `apply()`, creating structured and ready-to-merge records across all sources.

---

## 🔹 Step 4: Load and Merge Datasets

The loading system is designed to handle variations in column naming and structure across different CSV files. It works as follows:

- `load_dataset()` identifies relevant columns using aliases (e.g., "listed_in", "Genre" → "genre") and drops rows with missing critical fields.
- `load_and_merge()` applies the relevant normalization function and concatenates all valid entries into a single DataFrame for each use case.

This step results in three main DataFrames:

- `df_age`: Contains records with description and audience group, used for training age classification models.
- `df_genre`: Contains description and cleaned genre labels, used for genre prediction.
- `df_duration`: Includes numeric fields like duration, votes, and popularity, joined with audience group, for popularity analysis.

---

## 🔹 Step 5: Save Cleaned Datasets

Each processed dataset was saved as a CSV file:

- `audience_group_training_data.csv`
- `genre_training_data.csv`
- `duration_popularity_training_data.csv`

Each contains cleaned, normalized fields such as `description`, `normalized_genres`, `duration`, and `audience_group`.

---


##  Generated Datasets

The preprocessing script produces the following structured CSV files, each tailored for a specific task in the project:


| Filename                                | Description                                                                                      | How It Was Created                                                                                          | Purpose                                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `augmented_genre_only.csv`              | Augmented examples for multi-label genre classification.                                         | Texts were augmented using `nlpaug` for genres with relatively fewer samples.                               | To increase the representation of underrepresented genres during genre model training.        |
| `Decoded_Dataset_with_Genres_Sentiment_Violence.csv` | Dataset with genres, sentiment scores, and violence flags.                                       | Original dataset enriched with sentiment and violence features.                                              | Used as input for genre and audience classification models.                                  |
| `duration_popularity_training_data.csv` | Dataset containing duration and popularity-related fields.                                       | Extracted and cleaned from TMDb metadata.                                                                   | Used in modeling tasks related to content duration and popularity.                           |
| `multiinput_amazon_netflix_genre.csv`   | Main dataset with normalized genres and audience groups.                                         | Merged and preprocessed entries from Netflix and Amazon datasets.                                           | Used in all audience classification models.                                                   |
| `test_norm_titles.csv`                  | List of normalized titles for the test set.                                                      | Created during `train_test_split` to ensure dataset consistency.                                            | Used for consistent evaluation filtering.                                                     |
| `train_augmented_child.csv`             | Augmented training data for the `child` class only.                                              | Generated using `nlpaug` and saved separately.                                                              | Balances the training set without affecting validation/test sets.                             |



This improved model accuracy and reduced label noise.

---


## 🔹 Step 6: Genre Filtering

Some genres were removed after encoding due to ambiguity or low semantic value, such as:

- `"tv shows"`, `"movies"`, `"reality tv"`, `"international"`, `"cult"`

These were excluded using `clean_and_filter_genres()`.

Only the following genres were retained:

```
['action','sci-fi','comedy','documentary','drama','horror','thriller','romance']
```



![Image](https://github.com/user-attachments/assets/42d5a99f-6917-4a20-8701-67ef939ada43)



![Image](https://github.com/user-attachments/assets/a5c9efe9-be0a-46b8-9043-b438defeb652)

![Image](https://github.com/user-attachments/assets/e09dea57-d3d6-46c6-8324-98aab4d96dc5)


---

##  Final Outcome

The preprocessing pipeline lays the foundation for a modular system designed to **predict or generate missing metadata fields** in streaming content datasets. After cleaning, merging, and encoding:

- The datasets are structured to support three types of tasks:
  1. **Classification Models**:
     - Predict **genre** from `description` and `audience_group`.
     - Predict **audience_group** from `description` and `normalized_genres`.
  2. **Text Generation Model**:
     - Generate **description** from `title`, `normalized_genres`, and `audience_group` using text-based features.
  3. **Analytical Model**:
     - Analyze how **duration** relates to **popularity** across different audience segments.

- Each model is triggered **conditionally**, depending on which metadata field is missing.
- The system supports practical use cases such as:
  - **Metadata completion** for new content
  - **Improved catalog structuring**
  - **Personalized content tagging**

All models rely solely on **textual features**, in line with the project’s NLP focus.




# 🔎 Data Cleaning and Labeling Strategy

Initially, we identified the entries with softmax scores below the 0.5 threshold in the audience group. Upon further analysis, we observed that the **'teen'** feature often produced insufficient and misleading signals in the dataset. 

To address this:
- **Data labeling** and **active learning** techniques were applied for entries labeled as 'child' and 'adult'.
- Entries with **uncertain labels** were **removed** from the dataset to reduce noise.

  ![Image](https://github.com/user-attachments/assets/e388af69-e5c4-4f36-a02b-8d3d4aaf0378)

| Description                                                                                                                                                                                            | Audience | Popularity | Top Words (with Scores)                                                                                                                                                            |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Two Mississippi teens meet peculiar drifter Mud and get caught up in his web of tall tales about lost love, crimes of passion and bounty hunters.                                                       | teen     | 0.483      | passion (0.0840), crimes (-0.0761), web (-0.0471), teens (-0.0359), caught (0.0331), lost (0.0308), love (-0.0231), tales (0.0070), Mud (-0.0016), meet (-0.0014)                 |
| It's the treasure hunt of a lifetime for the Rescue Riders, who must race to find a precious golden dragon egg and protect it from evil pirates.                                                        | child    | 0.493      | Rescue (0.2240), hunt (-0.1683), race (0.1296), evil (0.0914), protect (-0.0601), Riders (-0.0020), from (0.0019), and (-0.0016), a (-0.0016), golden (0.0014)                   |
| Formerly a Korean resistance fighter, a policeman working for Japan is tasked with tracking down resistance leaders before they acquire explosives.                                                      | adult    | 0.501      | working (-0.0418), Korean (-0.0000), a (-0.0000), fighter (-0.0000), acquire (-0.0000), Formerly (-0.0000), Japan (-0.0000), resistance (-0.0000), explosives (-0.0000), leaders (-0.0000) |
| One of the most successful female groups in music history, TLC performs in concert while taking a look back at their tumultuous history.                                                                | child    | 0.418      | history (0.1822), successful (-0.0910), female (-0.0647), look (0.0212), music (0.0147), taking (-0.0046), TLC (0.0017), concert (0.0010), back (-0.0007), tumultuous (0.0006)   |
| In this short documentary, 37-year-old Cristina valiantly battles cancer while doing all she can to encourage others to live in the moment.                                                             | teen     | 0.424      | short (0.2491), cancer (-0.0505), year (-0.0348), old (0.0218), battles (-0.0214), documentary (0.0197), live (-0.0163), moment (-0.0015), 37 (0.0009), In (0.0008)              |
                                  |


---

# 🧠 Model Training

## ✅ TF-IDF + XGBoost

- **Textual inputs**: TF-IDF applied to `description` and `genre`
- **Numerical inputs**: `sentiment_score` and `violence_flag`
- **Augmented data**: Used **only in training** for the `child` class to avoid data leakage
- **Model parameters**:
  - Estimators: 250
  - Max depth: 5
  - Learning rate: 0.1
- **Evaluation**: Conducted on a separate test set using classification metrics and a confusion matrix
  ![Image](https://github.com/user-attachments/assets/df40497e-538a-4f4b-b89f-7f5f3c3dbe87)

---

## ✅ LSTM (GloVe + BiLSTM)

- Combined **GloVe embeddings** and **Bidirectional LSTM**
- Concatenated with:
  - TF-IDF genre features
  - Sentiment score
  - Violence flag
- **Augmented child data** used only during training
- **Class imbalance** handled with class weights
- **Early stopping** applied on validation set
- Final output used for:
  - Evaluation metrics
  - Ensemble input
 
![Image](https://github.com/user-attachments/assets/959a1434-f3b7-450b-a15b-6dd99897f5c7)
---

## ✅ DistilBERT

- Fine-tuned transformer model on `description` to classify audience group (`adult` / `child`)
- Data split into **train**, **validation**, and **test**
- **Augmented child data** included only in training
- Optimization via **Optuna**:
  - Learning rate
  - Dropout
  - Batch size
- **Weighted sampling** used for class imbalance
- Best model selected using **early stopping**
- Predictions saved for ensemble modeling

![Image](https://github.com/user-attachments/assets/39f4e780-56df-40a8-803e-5ea08de91e66)

---

# 🔗 Ensemble Model (Stacked Logistic Regression)

To enhance classification accuracy, an **ensemble model** was built by combining:

- DistilBERT
- GloVe-LSTM
- TF-IDF + XGBoost

### How It Works:
- Each model's **class probabilities + confidence scores** used as input features
- Meta-learner: **Logistic Regression (One-vs-Rest)**
- Trained on test set outputs of base models

![Image](https://github.com/user-attachments/assets/75356f75-4685-4174-acb6-c20cebdd595c)

### 📊 Final Evaluation:
- **Accuracy**, **macro F1-score**, and **confusion matrix** computed
 

| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Adult         | 0.86      | 0.82   | 0.84     | 652     |
| Child         | 0.59      | 0.66   | 0.62     | 256     |
| Accuracy      |           |        | 0.77     | 908     |
| Macro Avg     | 0.72      | 0.74   | 0.73     | 908     |
| Weighted Avg  | 0.78      | 0.77   | 0.78     | 908     |

**Overall Accuracy**: 0.7742  
**Macro F1 Score**: 0.7313


---

# 🎯 Multi-Label Genre Classification: Model Comparison

## 1. DistilBERT

| Metric      | Value (Macro Avg) |
|-------------|-------------------|
| Precision   | 0.70              |
| Recall      | 0.61              |
| F1-Score    | 0.65              |

**Highlights:**
- 📌 Documentary (F1: 0.83)
- 📌 Drama (F1: 0.71)
- 📌 Comedy (F1: 0.66)

---

## 2. BiLSTM + Attention

| Metric      | Value (Macro Avg) |
|-------------|-------------------|
| Precision   | 0.68              |
| Recall      | 0.57              |
| F1-Score    | 0.61              |

**Strengths:**
- 📌 Drama (F1: 0.68)
- 📌 Documentary (F1: 0.78)

---

## 3. TF-IDF + XGBoost

| Metric      | Value (Macro Avg) |
|-------------|-------------------|
| Precision   | 0.66              |
| Recall      | 0.54              |
| F1-Score    | 0.59              |

**Effective for:**
- 📌 Comedy (F1: 0.62)
- 📌 Documentary (F1: 0.75)

---

## 4. Ensemble Model (Stacked Logistic Regression)

| Metric      | Value (Macro Avg) |
|-------------|-------------------|
| Precision   | 0.71              |
| Recall      | 0.52              |
| F1-Score    | 0.59              |

**Highlights:**
- 📌 Documentary (F1: 0.86)
- Benefited genres: Comedy, Drama, Horror
- Fixed inconsistencies in Thriller and Sci-Fi
- Most balanced across labels

---

# ✅ Conclusion and Recommendation

- **DistilBERT** achieved the highest overall F1 due to its contextual understanding.
- **BiLSTM + Attention** was a solid alternative with moderate cost.
- **TF-IDF + XGBoost** was fast and interpretable.
- **Ensemble model** provided robustness and consistency, especially for noisy or imbalanced classes.

---

# 🔍 Final Macro F1-Score Summary

| Model                  | Macro F1 |
|------------------------|----------|
| DistilBERT             | 0.65     |
| BiLSTM + Attention     | 0.61     |
| TF-IDF + XGBoost       | 0.59     |
| Ensemble (Stacked LR)  | 0.59     |

📌 **Note**: Although Ensemble and XGBoost share the same macro F1 numerically, the ensemble yielded **better trade-offs** and **greater consistency** across genres.


## Analysis of Content Duration and Popularity
In our research, we examined whether there was a significant relationship between duration and popularity across different audience groups. The popularity index was set at a threshold of 7 where movies with a popularity score of 7 or higher were classified as popular and those with a score below 7 as not popular. We then performed both Spearman and Pearson correlation analyses to explore the relationship between movie duration and popularity. The results presented in below showed that the correlation coefficients in both analyses did not exceed 0.25 indicating a very weak correlation. Based on these findings, we conclude that there is no significant relationship between movie duration and popularity within the analyzed audience groups.



![Image](https://github.com/user-attachments/assets/28a337bb-67d7-4418-b9fa-61ac73312b9c)
![Image](https://github.com/user-attachments/assets/52bce09b-712a-43e1-a8f0-1e28e053f02c)

![Image](https://github.com/user-attachments/assets/f1471b5d-5e0a-4b02-9342-5c766a7e9b90)
---

## 🟢 How to Run

To run the project:

1. Make sure you have Python installed (version 3.8+ recommended).
2. Open a terminal or command prompt in the project folder.
3. Install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Pre-trained Embeddings

This project uses 100-dimensional GloVe embeddings. Please download the file manually:

- [Download GloVe 6B (Common Crawl)](https://nlp.stanford.edu/data/glove.6B.zip)

After downloading, extract and place the following file in a folder named `glove/` at the root of the project:



## 🧠 Ensemble Model Requirements

This project includes an ensemble model that combines the outputs of three individual classifiers:

- **DistilBERT model**
- **GloVe + LSTM model**
- **TF-IDF + XGBoost model**

Before running the ensemble script, make sure the following files are generated:

### 🎯 For Genre Classification:
```
outputs/genre/ensemble_input_distilbert_model.csv
outputs/genre/ensemble_input_glove_model.csv
outputs/genre/ensemble_input_xgboost_model.csv
```

### 🎯 For Audience Group Classification:
```
outputs/audience/ensemble_input_DistilBERT_model.csv
outputs/audience/ensemble_input_glove_model.csv
outputs/audience/ensemble_input_xgboost_model.csv
```
