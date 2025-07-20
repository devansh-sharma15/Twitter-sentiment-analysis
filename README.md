# Twitter-sentiment-analysis
🐦 Twitter Sentiment Analysis
This project implements a machine learning model to classify the sentiment of tweets as either positive or negative. It uses a large dataset of tweets and a Logistic Regression model to demonstrate the complete process—from data acquisition and preprocessing to model training, evaluation, and persistence.

📑 Table of Contents
✨ Features

📁 Dataset

⚙️ Installation

🚀 Usage

📊 Model Evaluation

📦 Dependencies

📝 License

✨ Features
**Data Acquisition**

Downloads the Sentiment140 dataset directly from Kaggle.

**Data Preprocessing**

Handles missing values (though none were found).

Converts target labels for binary classification (0 for negative, 1 for positive).

Cleans tweet text by removing non-alphabetic characters, converting to lowercase, removing stopwords, and applying Porter Stemming.

**Feature Extraction**

Uses TfidfVectorizer to convert text into numerical features.

**Machine Learning Model**

Trains a Logistic Regression model for binary sentiment classification.

**Model Evaluation**

Measures performance on both training and testing datasets.

**Model Persistence**

Saves the trained model using pickle for future use.

📁 **Dataset**
This project uses the Sentiment140 dataset, which contains 1.6 million tweets collected via the Twitter API. Each tweet is labeled as positive (4) or negative (0) based on emoticons. For this project, labels are converted as follows:

0 → Negative

4 → Positive (converted to 1)

⚙️ **Installation**
To run this project locally:

# Clone the repository:


git clone (https://github.com/devansh-sharma15/Twitter-sentiment-analysis.git)

# Install required libraries:

pip install numpy pandas scikit-learn nltk kaggle

# Set up Kaggle API Key:

Go to your Kaggle Account

Under API, click Create New API Token to download kaggle.json.

Place it in the appropriate directory:


mkdir -p ~/.kaggle
mv path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
🚀 **Usage**
# Open the notebook:

jupyter notebook "twetter_sentiment_analysis (2).ipynb"
# Run all the cells sequentially. The notebook will:

Download and extract the dataset

Preprocess the data

Train the Logistic Regression model

Evaluate performance

Save the trained model as trained_model.sav

📊 **Model Evaluation**
The trained Logistic Regression model achieved:

✅ **Training Accuracy**: ~83.56%

✅ **Test Accuracy**: ~77.08%

📦 **Dependencies**
numpy

pandas

re (built-in)

nltk

scikit-learn

kagglehub

zipfile (built-in)

pickle (built-in)

📝 License
This project is open-source and available under the MIT License.
