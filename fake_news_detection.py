
# Fake News Detection using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (downloaded from Kaggle or hosted online)
url = "https://raw.githubusercontent.com/sriharshavardhanchadaram/datasets/main/fake_or_real_news.csv"
df = pd.read_csv(url)

# Display dataset info
print("Dataset preview:\n", df.head())
print("\nLabel distribution:\n", df.label.value_counts())

# Define features and labels
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

# Initialize and train classifier
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(tfidf_train, y_train)

# Predict and evaluate
y_pred = model.predict(tfidf_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to make predictions on custom input
def predict_news(news_text):
    vect = tfidf.transform([news_text])
    prediction = model.predict(vect)
    return prediction[0]

# Example usage
sample_news = "NASA confirms the successful launch of a new space telescope."
print("\nSample Prediction:", predict_news(sample_news))
