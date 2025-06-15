# Multilingual-Language-Classifier
├── language_data/ # Contains one file per language (per split)
├── scripts/
│ ├── dataprocessing.py # Extracts 27 languages from MASSIVE and saves sentence files
│ ├── model_naive_bayes.py # Trains Naive Bayes classifier on 27 languages
│ ├── model_rda.py # Trains LDA/QDA classifiers on continent groupings
│ └── utils.py # Optional helper functions
├── notebooks/
│ └── assignment6.ipynb # End-to-end notebook (optional)
├── README.md
