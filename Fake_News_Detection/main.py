# Fake News Detection - Flask App with Styled HTML
# Developed by Ali Raza (BS AI Student)

# ======================
# 1. IMPORT LIBRARIES
# ======================
import pandas as pd
import numpy as np
import string
import re
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from flask import Flask, request, render_template

# ======================
# 2. NLTK DOWNLOAD
# ======================
nltk.download('stopwords')
nltk.download('wordnet')

# ======================
# 3. LOAD DATA
# ======================
df = pd.read_csv('news.csv')
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# ======================
# 4. TEXT CLEANING
# ======================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

df['clean_text'] = df['text'].apply(clean_text)

# ======================
# 5. TRAIN MODEL
# ======================
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# ======================
# 6. SAVE MODEL
# ======================
pickle.dump(model, open('nb_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

# ======================
# 7. FLASK APP
# ======================
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    cleaned = clean_text(text)
    vect = tfidf.transform([cleaned])
    prediction = model.predict(vect)[0]
    result = "REAL" if prediction == 1 else "FAKE"
    return render_template('index.html', prediction=result)

# ======================
# 8. STYLISH HTML TEMPLATE
# ======================
if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake News Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap" rel="stylesheet">
    <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    body {
      background-color: #0d1117;
      color: #eaeaea;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 40px;
    }

    .card {
      background-color: #161b22;
      width: 100%;
      max-width: 900px;
      padding: 50px 40px;
      border-radius: 16px;
      box-shadow: 0 0 25px rgba(0, 0, 0, 0.6);
      transition: all 0.3s ease;
    }

    .card:hover {
      transform: translateY(-5px);
      box-shadow: 0 0 40px rgba(0, 0, 0, 0.8);
    }

    h2 {
      text-align: center;
      font-size: 36px;
      font-weight: 700;
      background: linear-gradient(90deg, #00c6ff, #0072ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 30px;
    }

    textarea {
      width: 100%;
      height: 200px;
      background-color: #0d1117;
      color: #eaeaea;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 20px;
      font-size: 15px;
      resize: vertical;
      transition: border 0.3s;
    }

    textarea:focus {
      outline: none;
      border: 1px solid #58a6ff;
    }

    .btn {
      width: 100%;
      background: linear-gradient(135deg, #007bff, #00c6ff);
      color: white;
      padding: 14px;
      font-size: 18px;
      font-weight: 600;
      border: none;
      border-radius: 10px;
      margin-top: 25px;
      cursor: pointer;
      transition: background 0.3s ease, transform 0.2s;
    }

    .btn:hover {
      background: linear-gradient(135deg, #0056b3, #00aaff);
      transform: translateY(-2px);
    }

    .result-box {
      background-color: #1f6feb1a;
      border-left: 5px solid #58a6ff;
      padding: 18px;
      margin-top: 30px;
      border-radius: 6px;
      font-weight: 500;
      font-size: 16px;
    }

    .footer {
      text-align: center;
      margin-top: 35px;
      font-size: 14px;
      color: #8b949e;
      font-weight: bold;
      letter-spacing: 1px;
    }

    .footer span {
      color: #58a6ff;
      font-style: italic;
      text-shadow: 1px 1px 2px #007bff;
    }

    @media (max-width: 768px) {
      .card {
        padding: 35px 25px;
      }

      h2 {
        font-size: 28px;
      }

      .btn {
        font-size: 16px;
      }
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>🧠 Fake News Detection System</h2>
    <form method="POST" action="/predict">
      <textarea name="news" placeholder="Paste your news article here..."></textarea>
      <button type="submit" class="btn">🚀 Detect Now</button>
    </form>

    {% if prediction %}
    <div class="result-box">
      Prediction: <strong>{{ prediction }}</strong>
    </div>
    {% endif %}

    <div class="footer">Crafted with 💻 by <span>Ali Raza — BS AI Student</span></div>
  </div>
</body>
</html>
        """)
    app.run(debug=True)
