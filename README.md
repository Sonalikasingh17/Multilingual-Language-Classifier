# 🌐 Multilingual Language Classifier

[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://multilingual-language-classifier-ph3hen5kkyjtnvtpaxhmqy.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org)

 An end-to-end machine learning application for **multilingual language and continent classification** using the [MASSIVE dataset](https://huggingface.co/datasets/qanastek/MASSIVE) released by Amazon. This project achieves **96%+ accuracy** in language detection across 27 **Roman-script** languages spanning 4 continents.
 
Build classic machine learning models (Naive Bayes, LDA, QDA) for classification.The MASSIVE dataset contains parallel utterances across 51 languages. 

Focus on 27 **Roman-script languages**, build language-specific files from the MASSIVE dataset, and train two types of classifiers:
- **Multinomial Naive Bayes to classify 27 languages**
- **LDA/QDA (mimicking RDA) to classify sentences into 4 continent groups (Asia, Africa, Europe, North America)**

---

## 🎥 Demo

<div style="display: flex; flex-wrap: wrap; gap: 8px;">
  <img src="https://github.com/user-attachments/assets/faf7e739-85bd-4058-be74-f695e84727a1" alt="Screenshot 2025-08-22 005546" width="460"/>
  <img src="https://github.com/user-attachments/assets/3e3b8686-c3b9-46dd-95cc-561e4c67c101" alt="Screenshot 2025-08-22 005601" width="460"/>
  <img src="https://github.com/user-attachments/assets/861e8524-3493-41e8-b1f9-e9f90cfc4f35" alt="Screenshot 2025-08-22 005624" width="460"/>
  <img src="https://github.com/user-attachments/assets/7b5bbf33-d59c-449b-b8ad-c6aff005d39d" alt="Screenshot 2025-08-20 235157" width="460"/>
</div>

---

## ✨ Features

- 🎯 **High Accuracy**: 96.8% language classification, 89.4% continent classification
- 🌍 **27 Languages**: Support for Roman-script languages across 4 continents
- ⚡ **Real-time Prediction**: Instant classification with confidence scores
- 📊 **Batch Processing**: Analyze multiple texts simultaneously
- 🔄 **End-to-End Pipeline**: Complete MLOps workflow from data to deployment
- 🎨 **Interactive UI**: Beautiful Streamlit web interface

---

## 🌍 Supported Languages

- **Source**: [MASSIVE on Huggingface](https://huggingface.co/datasets/qanastek/MASSIVE)
- **Fields used**: `locale`, `utt`
- **Languages**:  
  ### Europe (19 languages)
  `da-DK` `de-DE` `es-ES` `fr-FR` `fi-FI` `hu-HU` `is-IS` `it-IT` `lv-LV` `nb-NO` `nl-NL` `pl-PL` `pt-PT` `ro-RO` `ru-RU` `sl-SL` `sv-SE` `sq-AL` `cy-GB`

  ### Asia (5 languages)
  `jv-ID` `ms-MY` `tl-PH` `tr-TR` `vi-VN`

  ### Africa (2 languages)
  `af-ZA` `sw-KE`

  ### North America (1 language)
  `en-US`

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multilingual-language-classifier.git
cd multilingual-language-classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models
```bash
python main.py train
```

### 5. Launch Web App
```bash
streamlit run app.py
```

---

## 📊 Performance Metrics

| Model | Task | Validation Accuracy | Test Accuracy |
|-------|------|------------|----------|
| Multinomial NB | Language Classification |  96.8% | 96.9% |
| LDA | Continent Classification |  89.4% | 89.0% |
| QDA | Continent Classification |  81.6% | 81.2% |

---

## 🏗️ Architecture

```
📥 Data Ingestion → 🔄 Data Transformation → 🤖 Model Training → 🚀 Deployment
```

---

## 🎓 Key Learnings & Technical Achievements

### **Data Engineering & Preprocessing**
- **Multilingual Dataset Handling**: Successfully processed the MASSIVE dataset containing 51 languages, focusing on 27 Roman-script languages across 4 continents
- **Efficient Data Loading**: Implemented optimized data loading pipelines that reduced processing time from hours to minutes using caching strategies
- **Cross-lingual Data Consistency**: Ensured data quality and consistency across multiple language variants and writing systems

### **Feature Engineering & Text Processing**
- **Character-level N-gram Analysis**: Implemented character-level TF-IDF vectorization (1-3 grams) for robust language identification, achieving superior performance over word-based approaches
- **Dimensionality Reduction**: Applied TruncatedSVD for efficient dimensionality reduction, reducing feature space from 15,000+ to 100 dimensions while maintaining classification accuracy
- **Hierarchical Label Mapping**: Developed sophisticated locale-to-continent mapping system for multi-level classification tasks

### **Machine Learning Model Development**
- **Classical ML Mastery**: Built and optimized multiple classical machine learning models:
  - **Multinomial Naive Bayes** for language classification (96.8% accuracy)
  - **Linear Discriminant Analysis (LDA)** for continent classification (89.4% accuracy)
  - **Quadratic Discriminant Analysis (QDA)** for alternative continent classification (81.6% accuracy)
- **Hyperparameter Optimization**: Fine-tuned model parameters including alpha values, feature limits, and n-gram ranges for optimal performance
- **Cross-validation Strategy**: Implemented robust train/validation/test splits ensuring reliable model evaluation

### **Advanced Analytics & Visualization**
- **Comprehensive Model Evaluation**: Created detailed classification reports, confusion matrices, and ROC curves for thorough performance analysis
- **Interactive Visualizations**: Developed multiple visualization types:
  - Sample distribution plots across languages
  - Confusion matrices with 27x27 language comparisons
  - ROC curves for multi-class classification
  - Feature importance analysis
- **Statistical Analysis**: Performed in-depth accuracy comparisons and model performance benchmarking

### **Production-Ready System Design**
- **End-to-End Pipeline Architecture**: Built complete MLOps pipeline from data ingestion to model deployment
- **Modular Code Design**: Implemented clean, maintainable code architecture with separate modules for:
  - Data ingestion and preprocessing
  - Model training and evaluation
  - Prediction pipelines
  - Web application interface
- **Error Handling & Logging**: Comprehensive error handling and logging system for production reliability

### **Deployment & User Experience**
- **Interactive Web Application**: Created user-friendly Streamlit interface with real-time predictions and confidence scores
- **Batch Processing Capabilities**: Implemented efficient batch processing for multiple text samples
- **Command-Line Interface**: Developed flexible CLI for various use cases and automation
- **Docker Containerization**: Containerized application for consistent deployment across environments

### **Performance Optimization**
- **Memory Efficiency**: Optimized memory usage for large-scale text processing
- **Processing Speed**: Achieved sub-second prediction times for single texts and efficient batch processing
- **Model Compression**: Implemented model serialization and compression for faster loading times

### **Research & Analysis Skills**
- **Comparative Model Analysis**: Conducted thorough comparison between different classical ML approaches
- **Language Similarity Analysis**: Investigated linguistic relationships through confusion matrix analysis
- **Cross-linguistic Pattern Recognition**: Identified patterns in character usage across different language families

---

## 🙏 Acknowledgements

Special thanks to Amazon Research for the [MASSIVE dataset](https://huggingface.co/datasets/qanastek/MASSIVE) and the open-source community behind scikit-learn, Streamlit, and Hugging Face. 

This project represents a journey in exploring classical machine learning approaches for multilingual text classification.


Made with ❤️ using Python and Streamlit
