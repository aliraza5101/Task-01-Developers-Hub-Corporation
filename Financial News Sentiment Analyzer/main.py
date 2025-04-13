# main.py
import pandas as pd
import string
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from flask import Flask, request, render_template
import os
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('financial_phrase_bank.csv', encoding='ISO-8859-1')
df.columns = ['label', 'headline']

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['headline'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Flask App
app = Flask(__name__)

# 🔥 Recent Prediction Tracker
recent_predictions = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    cleaned = clean_text(news)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]

    # 🔥 Track recent predictions (last 10)
    recent_predictions.append(pred)
    if len(recent_predictions) > 10:
        recent_predictions.pop(0)

    # 🔥 Chart data
    label_count = Counter(recent_predictions)
    chart_data = {
        'labels': list(label_count.keys()),
        'values': list(label_count.values())
    }

    return render_template('index.html', prediction=pred, chart_data=chart_data)

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Financial Sentiment Analyzer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-color: #0d0d0d;
      --card-bg: #1a1a1a;
      --accent-color: #ff4d4d;
      --text-color: #ffffff;
      --btn-gradient: linear-gradient(135deg, #ff4d4d, #ff1a1a);
    }

    body {
      font-family: 'Outfit', sans-serif;
      background: var(--bg-color);
      color: var(--text-color);
      transition: 0.3s ease;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
      padding-top: 80px; /* 🆕 space for fixed topbar */
    }

    .topbar {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  background: var(--card-bg);
  padding: 22px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 15px rgba(255, 77, 77, 0.1);
  z-index: 100;
}

/* 👇 Title thoda left se andar */
.title {
  font-size: 38px;
  font-weight: bold;
  background: linear-gradient(90deg, #ff4b2b, #ff416c);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-left: 100px;
}
.mid {
  font-size: 40px;
  font-weight: bold;
  background: #EFEFEF;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
/* 👇 Toggle thoda right se andar */
.toggle {
  font-size: 16px;
  background: #393939;
  padding: 10px 13px;
  border-radius: 50px;
  color: var(--text-color);
  cursor: pointer;
  transition: 0.3s;
  margin-right: 130px;
}

    .toggle:hover {
      background-color: #ff3b3b;
      color: #000;
    }

    .card {
      transform: scale(0.9);
      transform-origin: top center;
      transition: transform 0.3s ease;
      background: var(--card-bg);
      padding: 30px 40px;
      border-radius: 20px;
      width: 80%;
      max-width: 750px;
      box-shadow: 0 0 20px rgba(255, 77, 77, 0.2);
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    h1 {
      text-align: left;
      font-size: 45px;
      font-weight: 900;
      background: linear-gradient(90deg, #ff4b2b, #ff416c);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 27px;
    }

    .avatar {
      margin: 15px auto;
      padding-bottom: 15px;
      display: flex;
      justify-content: center;
    }

    .avatar img {
      width: 120px;
      border-radius: 50%;
      border: 0px solid var(--accent-color);
    }

    form {
      width: 90%;
    }

    textarea {
      width: 100%;
      padding: 15px;
      font-size: 16px;
      border-radius: 10px;
      border: none;
      resize: none;
      outline: none;
      background: rgba(255, 255, 255, 0.08);
      color: inherit;
      margin-bottom: 15px;
      box-sizing: border-box;
    }

    button {
      padding: 14px;
      font-size: 16px;
      font-weight: bold;
      background: var(--btn-gradient);
      border: none;
      border-radius: 12px;
      color: #fff;
      cursor: pointer;
      box-sizing: border-box;
      width: 100%;
    }

    .result {
      box-sizing: border-box;
      width: 90%;
      background: rgba(255, 77, 77, 0.15);
      border-left: 6px solid var(--accent-color);
      padding: 13px;
      border-radius: 12px;
      color: var(--accent-color);
      text-align: center;
      font-size: 17px;
      margin-top: 15px;
      margin-bottom: 0;
    }

    canvas {
      margin-top: 12px;
      width: 100%;
    }

    .footer {
      text-align: center;
      margin-top: 25px;
      font-size: 16px;
      font-weight: bold;
      color: #aaa;
      letter-spacing: 1px;
    }
  </style>
</head>
<body>
  <!-- 🆕 Toggle moved to fixed top bar -->
  <div class="topbar">
    <div class="title">🧠 Ali Raza</div>
    <div class="mid">AI/ML Internship</div>
    <div class="toggle" onclick="toggleMode()">🌗 Toggle Theme</div>
  </div>

  <div class="card">
    <h1>🧠 Financial Sentiment Analyzer</h1>

    <div class="avatar">
      <img src="{{ url_for('static', filename='my_avatar_2.jpg') }}" alt="AI Avatar" />
    </div>

    <form action="/predict" method="POST">
      <textarea name="news" rows="5" placeholder="Paste financial headline..."></textarea>
      <button type="submit">🔍 Analyze</button>
    </form>

    {% if prediction %}
      <div class="result">Predicted Sentiment: <strong>{{ prediction }}</strong></div>
    {% endif %}

    <div class="footer">Developed by Ali Raza ❤️ | BS AI Student</div>
  </div>

  <script>
    window.onload = function () {
      const savedTheme = localStorage.getItem("theme");
      if (savedTheme === "light") {
        setLightMode();
      } else {
        setDarkMode();
      }
    };

    function toggleMode() {
      const current = localStorage.getItem("theme");
      if (current === "light") {
        setDarkMode();
      } else {
        setLightMode();
      }
    }

    function setLightMode() {
      document.documentElement.style.setProperty('--bg-color', '#ffffff');
      document.documentElement.style.setProperty('--text-color', '#000000');
      localStorage.setItem("theme", "light");
    }

    function setDarkMode() {
      document.documentElement.style.setProperty('--bg-color', '#0d0d0d');
      document.documentElement.style.setProperty('--text-color', '#ffffff');
      localStorage.setItem("theme", "dark");
    }
  </script>
</body>
</html>


""")
    app.run(debug=True)
