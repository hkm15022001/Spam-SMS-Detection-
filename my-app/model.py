import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")

# Load your dataset
df = pd.read_csv('data.csv').rename(columns={'sms': 'text'})
df.head()

# Define a function to preprocess the text data
def preprocess_text(text):
    words = word_tokenize(text)  # Tokenization
    words = [word.lower() for word in words if word.isalnum()]  # Convert to lowercase
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(words)  # Concatenate tokens

# Apply the preprocessing function to the 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Split the data into features (X) and labels (y)
X = df['text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the Multinomial Naive Bayes classifier
sklearn_classifier = MultinomialNB(alpha=0.1)  # Alpha=0.1 is a smoothing parameter
sklearn_classifier.fit(X_train_tfidf, y_train)

# Define a custom NLTK classifier wrapper for the sklearn classifier
class SklearnNLTKClassifier(nltk.classify.ClassifierI):
    def __init__(self, classifier):
        self._classifier = classifier

    def classify(self, features):  # Predict for one feature
        return self._classifier.predict([features])[0]

    def classify_many(self, featuresets):  # Predict for multiple features
        return self._classifier.predict(featuresets)

    def prob_classify(self, features):  # Probability estimation not available
        raise NotImplementedError("Probability estimation not available.")

    def labels(self):  # Return labels
        return self._classifier.classes_

# Wrap the sklearn classifier with the NLTK interface
nltk_classifier = SklearnNLTKClassifier(sklearn_classifier)

# Use the classifier to predict labels for the test set
y_pred = nltk_classifier.classify_many(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
acc = f"Accuracy is : {accuracy:.2f}"

# Print the classification report and accuracy
print(report)
print(acc)
print("--------------------------------------------\n")
