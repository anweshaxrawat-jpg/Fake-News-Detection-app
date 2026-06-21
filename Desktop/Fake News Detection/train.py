import pandas as pd

print("Loading datasets...")

# International datasets

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

print("International datasets loaded")

# Indian dataset

indian = pd.read_csv("dataset/news_dataset.csv")
print("Indian dataset loaded")

# IFND dataset

ifnd = pd.read_csv("dataset/IFND.csv")

ifnd["label"] = ifnd["Label"].map({
"Fake": 0,
"TRUE": 1
})

ifnd["text"] = ifnd["Statement"]

print("IFND dataset loaded")

# WELFake dataset

welfake = pd.read_csv("dataset/WELFake_Dataset.csv")

welfake["title"] = welfake["title"].fillna("")
welfake["text"] = welfake["text"].fillna("")

welfake["text"] = welfake["title"] + " " + welfake["text"]

welfake = welfake[["text", "label"]]

# Convert labels to match project

# WELFake: 0=Real, 1=Fake

# Project: 0=Fake, 1=Real

welfake["label"] = welfake["label"].map({
0: 1,
1: 0
})

print("WELFake dataset loaded")

# Keep only required columns

fake = fake[["text", "label"]]
true = true[["text", "label"]]
ifnd = ifnd[["text", "label"]]

# Combine datasets

data = pd.concat(
[fake, true, ifnd, welfake],
ignore_index=True
)

# Shuffle data

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset Shuffled Successfully")

print("\nCombined Dataset Shape:")
print(data.shape)

print("\nChecking missing values...")
print(data.isnull().sum())

# Remove missing values

data = data.dropna()

print("\nMissing values removed")

from preprocess import clean_text

print("\nApplying text preprocessing...")

data["text"] = data["text"].apply(clean_text)

print("\nTotal articles:", len(data))

from sklearn.feature_extraction.text import TfidfVectorizer

print("\nApplying TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(
max_features=10000,
ngram_range=(1, 2),
stop_words="english",
min_df=2,
max_df=0.8
)

X = vectorizer.fit_transform(data["text"])
y = data["label"]

from sklearn.model_selection import train_test_split

print("\nSplitting dataset into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

from sklearn.linear_model import LogisticRegression

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
max_iter=1000,
random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib
import os

print("\nSaving model and vectorizer...")

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Saved successfully!")
