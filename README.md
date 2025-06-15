# Multilingual Language Classification (MASSIVE Dataset)

## Description

This repository contains the complete solution for DA5401 Assignment 6. The goal is to build classic machine learning models (Naive Bayes, LDA, QDA) for multilingual language and region classification using the [MASSIVE dataset](https://huggingface.co/datasets/qanastek/MASSIVE) released by Amazon.
The MASSIVE dataset contains parallel utterances across 51 languages. This project focuses on 27 **Roman-script** languages for a simplified language identification task.

We focus on 27 **Roman-script languages**, build language-specific files from the MASSIVE dataset, and train two types of classifiers:

- **Task 2**: Multinomial Naive Bayes to classify 27 languages
- **Task 3**: LDA/QDA (mimicking RDA) to classify sentences into 4 continent groups (Asia, Africa, Europe, North America)

---

##  Objective

- Extract and preprocess utterances for **27 Roman-script languages**
- Train a **Multinomial Naive Bayes** classifier for 27-language identification
- Group languages by continent and train classifiers to identify:
  - **Africa**
  - **Asia**
  - **Europe**
  - **North America**
- Use **LDA** and **QDA** for continent classification

---

##  Dataset Details

- **Source**: [MASSIVE on Huggingface](https://huggingface.co/datasets/qanastek/MASSIVE)
- **Fields used**: `locale`, `utt`
- **Languages**:  
  `af-ZA`, `da-DK`, `de-DE`, `en-US`, `es-ES`, `fr-FR`, `fi-FI`, `hu-HU`, `is-IS`, `it-IT`,  
  `jv-ID`, `lv-LV`, `ms-MY`, `nb-NO`, `nl-NL`, `pl-PL`, `pt-PT`, `ro-RO`, `ru-RU`, `sl-SL`,  
  `sv-SE`, `sq-AL`, `sw-KE`, `tl-PH`, `tr-TR`, `vi-VN`, `cy-GB`
  
---

##  Tasks

### Task 1: Extract Utterances
- Load and combine MASSIVE data across train, validation, and test splits
- Filters only **27 Roman-script locales** from the MASSIVE dataset.
- Output: consolidated `DataFrame` with `utt` and `locale`

---

### Task 2: Language Classification  (Multinomial Naive Bayes Classifier)
- **Model**: `TfidfVectorizer + MultinomialNB`
  (Trains a **Multinomial Naive Bayes** classifier using bag-of-words (`CountVectorizer`)).
- **Labels**: 27-language locales
- **Evaluation**: Accuracy + `classification_report` on validation and test sets

---

### Task 3: Continent Classification (LDA & QDA Classifiers)
- **Mapping**: Locale → Country → Continent
  Maps the 27 languages to 4 continent classes:
  - **Asia**: ms-MY, jv-ID, tl-PH, tr-TR, vi-VN
  - **Africa**: af-ZA, sw-KE
  - **Europe**: cy-GB, da-DK, de-DE, ... (most of them)
  - **North America**: en-US
- **Model**: `TfidfVectorizer + TruncatedSVD + LDA/QDA`
  (Converts sentences to TF-IDF vectors and reduces dimensionality using `TruncatedSVD`).
- **Labels**: Africa, Asia, Europe, North America
- **Evaluation**: Classification report + Accuracy

---

##  Results

### Language Classification (Multinomial Naive Bayes)

| Metric     | Validation | Test     |
|------------|------------|----------|
| Accuracy   | ~98.4%     | ~98.3%   |

### Continent Classification

| Model | Validation Accuracy | Test Accuracy |
|-------|---------------------|---------------|
| LDA   | ~89.9%              | ~89.6%        |
| QDA   | ~69.3%              | ~79.1%        |

---

##  Key Learnings

- How to preprocess multilingual NLP data
- Building classic text classification pipelines (TF-IDF + NB)
- Label transformation for hierarchical tasks (language → region)
- Dimensionality reduction using TruncatedSVD
- LDA vs QDA performance in NLP tasks

---

##  Usage

### Clone Repository
```bash
git clone https://github.com/Sonalikasingh17/Multilingual-Language-Classifier.git
cd Multilingual-Language-Classifier
```


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

- [MASSIVE Dataset on Huggingface](https://huggingface.co/datasets/qanastek/MASSIVE)
- [scikit-learn documentation](https://scikit-learn.org/stable/)

---
## Author
 Sonalika Singh
 
 IIT Madras


