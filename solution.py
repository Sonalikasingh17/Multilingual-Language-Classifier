# DA5401 Assignment 6 – Multilingual Language Classification

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.pipeline import make_pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Setup directory and languages
massive_dataset = load_dataset("qanastek/MASSIVE", "en-US", split='test', trust_remote_code=True)
print(massive_dataset)
print(massive_dataset[0])

languages = [
    'af-ZA', 'da-DK', 'de-DE', 'en-US', 'es-ES', 'fr-FR', 'fi-FI', 'hu-HU', 'is-IS', 'it-IT',
    'jv-ID', 'lv-LV', 'ms-MY', 'nb-NO', 'nl-NL', 'pl-PL', 'pt-PT', 'ro-RO', 'ru-RU', 'sl-SL',
    'sv-SE', 'sq-AL', 'sw-KE', 'tl-PH', 'tr-TR', 'vi-VN', 'cy-GB'
]

continent_lookup = {
    'ZA': 'Africa', 'KE': 'Africa', 'AL': 'Europe', 'GB': 'Europe', 'DK': 'Europe', 'DE': 'Europe',
    'ES': 'Europe', 'FR': 'Europe', 'FI': 'Europe', 'HU': 'Europe', 'IS': 'Europe', 'IT': 'Europe',
    'ID': 'Asia', 'LV': 'Europe', 'MY': 'Asia', 'NO': 'Europe', 'NL': 'Europe', 'PL': 'Europe',
    'PT': 'Europe', 'RO': 'Europe', 'RU': 'Europe', 'SL': 'Europe', 'SE': 'Europe', 'PH': 'Asia',
    'TR': 'Asia', 'VN': 'Asia', 'US': 'North America'
}

# Step 2: Load all splits

def load_massive_split(langs, split):
    all_data = []
    for lang in langs:
        ds = load_dataset("qanastek/MASSIVE", lang, split=split, trust_remote_code=True)
        df = pd.DataFrame(ds)
        df = df[['locale', 'utt']].copy()
        df['split'] = split
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

train_df = load_massive_split(languages, 'train')
val_df = load_massive_split(languages, 'validation')
test_df = load_massive_split(languages, 'test')

# Step 3: Train language classifier
X_train = train_df['utt']
y_train = train_df['locale']

X_val = val_df['utt']
y_val = val_df['locale']

X_test = test_df['utt']
y_test = test_df['locale']

pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
pipeline.fit(X_train, y_train)

# Step 4: Evaluate language model
val_preds = pipeline.predict(X_val)
test_preds = pipeline.predict(X_test)

print("\n--- Naive Bayes Language Classification ---")
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Validation Report:")
print(classification_report(y_val, val_preds))
print("Test Report:")
print(classification_report(y_test, test_preds))

# Step 5: Add continent labels
def extract_country(locale):
    return locale.split('-')[1]

def map_continent(locale):
    country = extract_country(locale)
    return continent_lookup.get(country, 'Unknown')

train_df['continent'] = train_df['locale'].apply(map_continent)
val_df['continent'] = val_df['locale'].apply(map_continent)
test_df['continent'] = test_df['locale'].apply(map_continent)

# Step 6: Train LDA/QDA for continent classification
X_train = train_df['utt']
y_train = train_df['continent']
X_val = val_df['utt']
y_val = val_df['continent']
X_test = test_df['utt']
y_test = test_df['continent']

# Vectorize and reduce
vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_val_vec = vec.transform(X_val)
X_test_vec = vec.transform(X_test)

svd = TruncatedSVD(n_components=100)
X_train_red = svd.fit_transform(X_train_vec)
X_val_red = svd.transform(X_val_vec)
X_test_red = svd.transform(X_test_vec)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_red, y_train)
val_lda = lda.predict(X_val_red)
test_lda = lda.predict(X_test_red)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_red, y_train)
val_qda = qda.predict(X_val_red)
test_qda = qda.predict(X_test_red)

print("\n--- LDA Continent Classification ---")
print("Validation Accuracy:", accuracy_score(y_val, val_lda))
print("Test Accuracy:", accuracy_score(y_test, test_lda))
print(classification_report(y_val, val_lda))

print("\n--- QDA Continent Classification ---")
print("Validation Accuracy:", accuracy_score(y_val, val_qda))
print("Test Accuracy:", accuracy_score(y_test, test_qda))
print(classification_report(y_test, test_qda))


# Step 7: Save the models and vectorizer
import joblib
joblib.dump(pipeline, 'language_classifier.pkl')
joblib.dump(vec, 'tfidf_vectorizer.pkl')
joblib.dump(svd, 'svd_reducer.pkl')
joblib.dump(lda, 'lda_model.pkl')
joblib.dump(qda, 'qda_model.pkl')
# Step 8: Save the dataframes
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
# Save the continent mappings
joblib.dump(continent_lookup, 'continent_lookup.pkl')
# Save the language mappings
language_lookup = {lang: lang.split('-')[0] for lang in languages}
joblib.dump(language_lookup, 'language_lookup.pkl')
# Save the language and continent mappings
language_continent_lookup = {lang: map_continent(lang) for lang in languages}
joblib.dump(language_continent_lookup, 'language_continent_lookup.pkl')
# Step 9: Print completion message
print("Models and data saved successfully. You can now use the saved models for predictions.")
# Step 10: Load and test saved models
def load_and_test_models():
    # Load the models and vectorizer
    language_classifier = joblib.load('language_classifier.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    svd_reducer = joblib.load('svd_reducer.pkl')
    lda_model = joblib.load('lda_model.pkl')
    qda_model = joblib.load('qda_model.pkl')

    # Load the dataframes
    train_data = pd.read_csv('train_df.csv')
    val_data = pd.read_csv('val_data.csv')
    test_data = pd.read_csv('test_data.csv')

    # Test the language classifier
    sample_texts = ["Hello, how are you?", "Bonjour, comment ça va?", "Hola, ¿cómo estás?"]
    sample_preds = language_classifier.predict(sample_texts)
    print("Sample Predictions for Language Classifier:", sample_preds)
    # Test the continent classifier
    sample_texts_vec = tfidf_vectorizer.transform(sample_texts)
    sample_texts_red = svd_reducer.transform(sample_texts_vec)
    lda_preds = lda_model.predict(sample_texts_red)
    qda_preds = qda_model.predict(sample_texts_red)
    print("Sample Predictions for LDA Continent Classifier:", lda_preds)
    print("Sample Predictions for QDA Continent Classifier:", qda_preds)
    return sample_preds, lda_preds, qda_preds
# Load and test the saved models
# load_and_test_models()
# Uncomment the line below to test the saved models
# load_and_test_models()
# Step 11: Finalize the script
if __name__ == "__main__":
    print("Script executed successfully. All models and data are saved.")
   

# Uncomment the line below to run the load and test function
    # load_and_test_models() 

# This script is now complete and ready for use in multilingual language classification tasks.

   
