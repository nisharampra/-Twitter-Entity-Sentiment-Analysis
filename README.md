# -Twitter-Entity-Sentiment-Analysis

# 🐦 Twitter Entity Sentiment Classification

A Natural Language Processing (NLP) project to classify the sentiment of tweets directed toward specific entities (e.g., brands like Amazon or Microsoft). This project compares statistical and embedding-based models for text classification.

## 📌 Problem Domain

Twitter is full of short, noisy, opinionated content. This project investigates how well different models handle such data by identifying sentiments expressed in tweets toward entities.

## 📊 Objectives

- Build and evaluate two sentiment classification models:
  - **Naive Bayes with Bag-of-Words**
  - **Logistic Regression with Word2Vec/TF-IDF**
- Compare performance using metrics like precision, recall, and F1-score.
- Visualize class distribution and performance comparisons.

## 🧾 Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Fields**:
  - `ID` – Tweet ID
  - `Entity` – Brand mentioned
  - `Sentiment` – One of: Positive, Negative, Neutral, Irrelevant
  - `Tweet` – Raw tweet text

## ⚙️ Preprocessing

- Lowercasing
- Removing URLs, mentions, hashtags, and punctuation
- Stopword filtering

## 🧠 Models Used

| Model Type | Features           | Notes                              |
|------------|--------------------|-------------------------------------|
| Naive Bayes | Bag-of-Words       | Simple, fast, decent for text tasks |
| Logistic Regression | Word2Vec / TF-IDF | Captures semantic patterns          |

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 📎 Example Code Snippet

```python
def preprocess(text):
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()
    return " ".join([t for t in tokens if t not in custom_stopwords])
