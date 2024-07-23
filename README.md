# Medical Text Classification

> This project explores healthcare clinical data (transcription/physician notes) and develops a classifier to categorize transcription notes (clinical notes) into medical specialties using machine learning (ML) and deep learning (DL) techniques.

## Data Preprocessing
- Transcription notes (medical text data) are converted into a frequency word dictionary for each clinical note and then transformed into a data frame using the `freq_words` function.

## Feature Engineering
- Utilized the TF-IDF vectorizer to convert each clinical note into a vector feature space, as tfidf is optimal for classification technique.

