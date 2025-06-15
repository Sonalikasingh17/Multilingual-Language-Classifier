# Multilingual Language Classification (MASSIVE Dataset)

## Description

This repository contains the complete solution for DA5401 Assignment 6. The goal is to build classic machine learning models (Naive Bayes, LDA, QDA) for multilingual language and region classification using the [MASSIVE dataset](https://huggingface.co/datasets/qanastek/MASSIVE) released by Amazon.
The MASSIVE dataset contains parallel utterances across 51 languages. This project focuses on 27 **Roman-script** languages for a simplified language identification task.

We focus on 27 **Roman-script languages**, build language-specific files from the MASSIVE dataset, and train two types of classifiers:

- **Task 2**: Multinomial Naive Bayes to classify 27 languages
- **Task 3**: LDA/QDA (mimicking RDA) to classify sentences into 4 continent groups (Asia, Africa, Europe, North America)

---

##  Tasks

###  Task 1: Extract Sentences from MASSIVE

- Filters only **27 Roman-script locales** from the MASSIVE dataset.
- Creates separate `.txt` files for `train`, `validation`, and `test` sets for each locale.
- Each file contains one sentence (`utt`) per line.

###  Task 2: Multinomial Naive Bayes Classifier

- Trains a **Multinomial Naive Bayes** classifier using bag-of-words (`CountVectorizer`).
- Predicts one of the 27 language locales.
- Evaluates model on `train`, `validation`, and `test` partitions using `classification_report`.

###  Task 3: LDA & QDA Classifiers for Continent Prediction

- Maps the 27 languages to 4 continent classes:
  - **Asia**: ms-MY, jv-ID, tl-PH, tr-TR, vi-VN
  - **Africa**: af-ZA, sw-KE
  - **Europe**: cy-GB, da-DK, de-DE, ... (most of them)
  - **North America**: en-US
- Converts sentences to TF-IDF vectors and reduces dimensionality using `TruncatedSVD`.
- Trains **LDA** and **QDA** (as proxies for RDA).
- Evaluates predictions for both models.

---

## How to Run
Run the script
``` yaml
python solution.py
```
## Learnings
- Working with real-world multilingual datasets
- Preprocessing and tokenization
- Building simple classifiers for language identification
- Dimensionality reduction and classical discriminant analysis
  
---

## References
- MASSIVE Dataset on Huggingface
- scikit-learn documentation

---
## Author
Sonalika Singh
IIT Madras


