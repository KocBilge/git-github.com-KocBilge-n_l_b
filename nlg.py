import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')

# Sample data set
data = [
    ("Bu film çok harika!", "olumlu"),
    ("Berbat bir deneyimdi.", "olumsuz"),
    ("Harika bir restorandı.", "olumlu"),
    ("Bu ürün tam bir hayal kırıklığı.", "olumsuz"),
]

# Sorting data into features and tags
corpus = [text for text, label in data]
labels = [label for text, label in data]

# Dividing data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size=0.2, random_state=42)

# Create a function for text processing
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('turkish'))  # Türkçe için stop words
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Convert text to feature vectors
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Training the Naive Bayes model
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Evaluate the model on the test set
y_pred = classifier.predict(X_test_vectors)

# Showing results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predicting a sample text
sample_text = "This product is perfect!"
sample_text_vectorized = vectorizer.transform([preprocess_text(sample_text)])
prediction = classifier.predict(sample_text_vectorized)[0]
print(f"\nÖrnek metin: '{sample_text}'\nPredict: {prediction}")
