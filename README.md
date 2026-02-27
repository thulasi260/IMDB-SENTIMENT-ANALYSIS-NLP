🎬 IMDb Sentiment Analysis using NLP & Machine Learning
📌 Project Overview
This project builds an end-to-end Natural Language Processing (NLP) pipeline to classify IMDb movie reviews as Positive or Negative using Machine Learning models.
The objective is to demonstrate text preprocessing, feature engineering using TF-IDF, model comparison, and performance evaluation on real-world review data.

📂 Dataset
IMDb Movie Review Dataset
Real-world movie reviews
Binary classification (Positive / Negative)
4,000 reviews sampled for training and evaluation
This dataset contains naturally occurring noisy text including punctuation, HTML tags, and varying sentence structures.

🧹 Data Preprocessing
Text cleaning steps:
Converted text to lowercase
Removed HTML tags
Removed special characters and numbers
Removed English stopwords using NLTK
Tokenized and reconstructed cleaned text
Proper preprocessing improves model performance by reducing noise.

🔎 Feature Engineering
Used TF-IDF (Term Frequency – Inverse Document Frequency) to convert text into numerical vectors.
Configuration:
max_features = 5000
ngram_range = (1,2) (Unigrams + Bigrams)
Bigrams help capture contextual patterns like:
“not good”
“very bad”

🤖 Models Implemented
Two classification models were trained and compared:
Logistic Regression
Multinomial Naive Bayes
Both models were trained using an 80-20 train-test split.

📊 Model Performance
Model	Test Accuracy
Logistic Regression	~85.25%
Naive Bayes	~85.12%

Cross-Validation
5-Fold Cross Validation Average Accuracy: ~84.3%
The close alignment between test accuracy and cross-validation indicates:
Good generalization
Minimal overfitting
Stable model performance

📈 Evaluation Metrics
Accuracy Score
Confusion Matrix
Cross-Validation

🧠 Key Learnings
Handling real-world noisy text data
Implementing TF-IDF with n-grams
Comparing multiple ML models
Evaluating stability using cross-validation
Understanding limitations of bag-of-words models

⚠ Limitations
TF-IDF treats words independently
Cannot fully understand contextual meaning or sarcasm
Mixed sentiment sentences may be misclassified

🔮 Future Improvements
Hyperparameter tuning
Support Vector Machine (SVM)
Deep Learning models (LSTM / BERT)
Model deployment using Streamlit

🛠 Tech Stack
Python
NLTK
Scikit-learn
Pandas
Matplotlib / Seaborn

🚀 Outcome

This project demonstrates a complete NLP workflow from preprocessing to evaluation using real-world data, achieving strong baseline performance (~85% accuracy).
