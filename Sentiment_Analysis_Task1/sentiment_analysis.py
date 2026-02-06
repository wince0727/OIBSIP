import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

twitter_df = pd.read_csv("Twitter_Data.csv")
print(twitter_df.head())

twitter_df = twitter_df.dropna(subset=['category'])

twitter_df = twitter_df.reset_index(drop=True)

print("After removing NaN labels:")
print(twitter_df.isna().sum())

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

twitter_df['clean_text'] = twitter_df['clean_text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(twitter_df['clean_text'])
y = twitter_df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")   
plt.show()


twitter_df['category'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.savefig("sentiment_distribution.png")   
plt.show()

sample = ["This product is very good and I like it"]
sample_clean = [clean_text(sample[0])]
sample_vector = vectorizer.transform(sample_clean)
prediction = model.predict(sample_vector)

print("Predicted Sentiment:", prediction)
