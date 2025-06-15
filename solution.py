# DA5401 Assignment 6: Multilingual Language Classification using MASSIVE

import os
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from collections import defaultdict


# Load dataset
dataset = load_dataset("qanastek/MASSIVE", trust_remote_code=True)

# Create output directory
os.makedirs("language_data", exist_ok=True)

# Define Roman-script locales
locales = [
    'af-ZA', 'da-DK', 'de-DE', 'en-US', 'es-ES', 'fr-FR', 'fi-FI', 'hu-HU', 'is-IS', 'it-IT',
    'jv-ID', 'lv-LV', 'ms-MY', 'nb-NO', 'nl-NL', 'pl-PL', 'pt-PT', 'ro-RO', 'ru-RU', 'sl-SL',
    'sv-SE', 'sq-AL', 'sw-KE', 'tl-PH', 'tr-TR', 'vi-VN', 'cy-GB'
]

# Locale to continent mapping
continent_map = {
    'af-ZA': 'Africa', 'sw-KE': 'Africa',
    'cy-GB': 'Europe', 'da-DK': 'Europe', 'de-DE': 'Europe', 'es-ES': 'Europe', 'fi-FI': 'Europe',
    'fr-FR': 'Europe', 'hu-HU': 'Europe', 'is-IS': 'Europe', 'it-IT': 'Europe', 'lv-LV': 'Europe',
    'nb-NO': 'Europe', 'nl-NL': 'Europe', 'pl-PL': 'Europe', 'pt-PT': 'Europe', 'ro-RO': 'Europe',
    'ru-RU': 'Europe', 'sl-SL': 'Europe', 'sq-AL': 'Europe', 'sv-SE': 'Europe',
    'en-US': 'North America',
    'ms-MY': 'Asia', 'jv-ID': 'Asia', 'tl-PH': 'Asia', 'tr-TR': 'Asia', 'vi-VN': 'Asia'
}

# Load MASSIVE dataset
print("Loading dataset...")
dataset = load_dataset("qanastek/MASSIVE")

# Save utterances per locale and partition
def save_utts(locale, data, part):
    utts = [sample['utt'] for sample in data if sample['locale'] == locale]
    with open(f"language_data/{locale}_{part}.txt", "w", encoding="utf-8") as f:
        f.write('\n'.join(utts))

print("Extracting sentences for each locale...")
for part in ['train', 'validation', 'test']:
    data = dataset[part]
    for locale in locales:
        save_utts(locale, data, part)

# Load text and labels
def load_texts(part):
    texts, labels = [], []
    for locale in locales:
        file = f"language_data/{locale}_{part}.txt"
        with open(file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            texts.extend(lines)
            labels.extend([locale] * len(lines))
    return texts, labels

print("Loading text data for training...")
X_train, y_train = load_texts('train')
X_val, y_val = load_texts('validation')
X_test, y_test = load_texts('test')

# --- Task 2: Naive Bayes ---
print("Training Multinomial Naive Bayes classifier...")
vectorizer_nb = CountVectorizer()
X_train_vec_nb = vectorizer_nb.fit_transform(X_train)
X_val_vec_nb = vectorizer_nb.transform(X_val)
X_test_vec_nb = vectorizer_nb.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_vec_nb, y_train)

print("\n--- Naive Bayes Results ---")
print("Train:")
print(classification_report(y_train, nb.predict(X_train_vec_nb)))
print("Validation:")
print(classification_report(y_val, nb.predict(X_val_vec_nb)))
print("Test:")
print(classification_report(y_test, nb.predict(X_test_vec_nb)))

# --- Task 3: RDA / LDA / QDA ---
print("\nTraining LDA and QDA classifiers on continent labels...")

def get_group_labels(label_list):
    return [continent_map[loc] for loc in label_list]

y_train_group = get_group_labels(y_train)
y_val_group = get_group_labels(y_val)
y_test_group = get_group_labels(y_test)

vectorizer_rda = TfidfVectorizer(min_df=2, max_df=0.95)
X_train_vec_rda = vectorizer_rda.fit_transform(X_train)
X_val_vec_rda = vectorizer_rda.transform(X_val)
X_test_vec_rda = vectorizer_rda.transform(X_test)

# Reduce dimensions for LDA/QDA
svd = TruncatedSVD(n_components=100)
X_train_reduced = svd.fit_transform(X_train_vec_rda)
X_val_reduced = svd.transform(X_val_vec_rda)
X_test_reduced = svd.transform(X_test_vec_rda)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_reduced, y_train_group)
print("\n--- LDA Results ---")
print("Validation:")
print(classification_report(y_val_group, lda.predict(X_val_reduced)))
print("Test:")
print(classification_report(y_test_group, lda.predict(X_test_reduced)))

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_reduced, y_train_group)
print("\n--- QDA Results ---")
print("Validation:")
print(classification_report(y_val_group, qda.predict(X_val_reduced)))
print("Test:")
print(classification_report(y_test_group, qda.predict(X_test_reduced)))

print("\nAll tasks completed successfully.")
