# Medical Text Classification

> This project explores healthcare clinical data (transcription/physician notes) and develops a classifier to categorize transcription notes (clinical notes) into medical specialties using machine learning (ML) and deep learning (DL) techniques.

## SETUP
    Upload files to google drive
    Connect google drive to colab notebook
    Retrieve files in list using glob library
    

## Data Preprocessing
- Data Cleaning
   + Removing punctutation
   + Converting all characters to lowercase
   + Removing specific pattern using Regex (measurement details, numeric information)
   + Removing punctuation
   + Removing Non-English Words
   + Removing Stopwords
   + Removing Patient identifiers using NER
   + Tokenizing each word
   + Lemmatization of spacy tokenizer
   + Creating frequency of words dictionary of a document

- Transcription notes (medical text data) are converted into a frequency word dictionary for each clinical note and then transformed into a data frame using the `freq_words` function.

## Feature Engineering
- Utilized the TF-IDF vectorizer to convert each clinical note into a vector feature space, as `TF_IDF` is optimal for classification technique.
- Methods For Feature Selection
  + Univariate feature selection method - Done using SELECTKBEST() that selects top k features based on the specified scoring function. It works by evaluating each column individually using a statistical test and selects top k features with the highest scores

  + Feature Importance method

## Modeling
- Modeled the training process by splitting the dataset into 80% training, 10% validation, and 10% test sets. We utilized GridSearchCV for hyperparameter tuning and found that the SVM
  model outperformed other models, such as Random Forest, Decision Tree, and Logistic Regression, with an accuracy of 87%.

- Additionally, a 95% boost analysis was conducted by comparing baseline productivity metrics before and after implementing NLP pipelines. This analysis included time taken for
  preprocessing, error rates, and computational resource usage.




