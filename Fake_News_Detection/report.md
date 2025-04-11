✅ 1. Report for Internship Submission (Developers HUB Corporation)
Title: Fake News Detection using Machine Learning

Name: Ali Raza
Program: BS Artificial Intelligence
Internship Task: Task 1 – Fake News Detection System
Organization: Developers HUB Corporation

1. Introduction
In today's digital era, fake news spreads rapidly across various platforms, misleading millions of people. The goal of this task was to create a machine learning-based system that can detect whether a news article is FAKE or REAL.

2. Approach Used
● Dataset:
We used the Fake and Real News Dataset, which includes two classes: FAKE and REAL.

● Preprocessing Steps:
To ensure clean and useful data for model training, the following steps were performed:

Conversion to lowercase

Removal of digits and punctuation

Tokenization and stopword removal

Lemmatization (using WordNet Lemmatizer)

● Model Training:
Out of the three options given, the Naïve Bayes Classifier was implemented using the following techniques:

TfidfVectorizer to convert text into numeric features

Multinomial Naive Bayes for classification

● Web Deployment:
A stylish and responsive Flask web app was developed, allowing users to:

Enter a news article in a textbox

Click a button to detect whether the article is REAL or FAKE

Get instant results on the same page

3. Challenges Faced
Handling text noise like special characters and different cases

Ensuring efficient preprocessing for better model performance

Integrating the machine learning model with a Flask app

Styling the HTML to give a professional and user-friendly UI

4. Model Performance & Improvements
The model showed good accuracy on the test set (as observed from accuracy_score and classification report).

Future improvements could include:

Using advanced models like Random Forest and LSTM

Adding real-time news source validation

Expanding dataset for better generalization

5. Conclusion
This task helped me understand the entire pipeline of an ML project — from data cleaning to deployment. I also learned how to make AI solutions accessible via web apps.

