import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
import re

# Initialize the stemmer
stemmer = PorterStemmer()

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return ' '.join([stemmer.stem(token) for token in tokens])

# Dataset
data = {
    'garbage': [
        "nuculear waste", "steel", "Liquids", "wood", "animal_waste", "glass",
        "bones", "plastic", "fuel", "container", "oil", "bromine", "honey",
        "electronic", "paints", "polluted_water", "wool", "sewage"
    ],
    'label': [
        'wet', 'dry', 'wet', 'dry', 'wet', 'dry',
        'dry', 'dry', 'wet', 'dry', 'wet', 'wet',
        'wet', 'dry', 'wet', 'wet', 'dry', 'wet'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Apply preprocessing
df['cleaned'] = df['garbage'].apply(preprocess)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict new input
new_input = ["glasses"]
new_input_cleaned = [preprocess(x) for x in new_input]
new_vector = vectorizer.transform(new_input_cleaned)
prediction = model.predict(new_vector)
print(f"Prediction for '{new_input[0]}': {prediction[0]}")






