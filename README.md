# Twitter-sentiment-analysis
Twitter Sentiment Analysis
This project implements a machine learning model to classify the sentiment of tweets as either positive or negative. It leverages a large dataset of tweets to train a Logistic Regression model, demonstrating the process from data acquisition and preprocessing to model training and evaluation.

Table of Contents
Features

Dataset

Installation

Usage

Model Evaluation

Dependencies

License

Features
Data Acquisition: Downloads the Sentiment140 dataset directly from Kaggle.

Data Preprocessing:

Handles missing values (though none were found in this dataset).

Converts target labels for binary classification (0 for negative, 1 for positive).

Performs text cleaning including removing non-alphabetic characters, converting to lowercase, and removing stopwords.

Applies Porter Stemming to reduce words to their root form.

Feature Extraction: Utilizes TfidfVectorizer to convert textual data into numerical features.

Machine Learning Model: Employs a Logistic Regression model for sentiment classification.

Model Evaluation: Calculates accuracy scores on both training and test datasets.

Model Persistence: Saves the trained model using pickle for future use.

Dataset
The project uses the Sentiment140 dataset which contains 1.6 million tweets extracted from the Twitter API. The tweets have been pre-classified as positive (4) or negative (0) based on the presence of emoticons. In this project, the '4' target is converted to '1' for binary classification.

Installation
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

(Replace your-username and your-repo-name with your actual GitHub details.)

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install the required libraries:

pip install numpy pandas scikit-learn nltk kaggle

Kaggle API Key:
To download the dataset, you need a Kaggle API key.

Go to your Kaggle account profile (https://www.kaggle.com/<your-username>/account).

Under the "API" section, click "Create New API Token". This will download a kaggle.json file.

Place this kaggle.json file in the ~/.kaggle/ directory on your system. If the directory doesn't exist, create it:

mkdir -p ~/.kaggle
mv path/to/your/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

Usage
Open the Jupyter Notebook:

jupyter notebook "twetter_sentiment_analysis (2).ipynb"

Run all cells: Execute all cells in the notebook sequentially. The notebook will:

Download and extract the dataset.

Perform data preprocessing.

Train the Logistic Regression model.

Evaluate the model's performance.

Save the trained model as trained_model.sav.

Model Evaluation
The Logistic Regression model achieved the following accuracy scores:

Accuracy on training data: approximately 83.56%

Accuracy on test data: approximately 77.08%

Dependencies
numpy

pandas

re (built-in)

nltk (Natural Language Toolkit)

sklearn (Scikit-learn)

kagglehub

zipfile (built-in)

pickle (built-in)

License
This project is open-source and available under the MIT License.
