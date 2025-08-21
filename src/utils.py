import os
import sys
import requests
import joblib
import pickle
import yaml
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import get_logger


logger = get_logger(__name__)

def download_from_kaggle():
    import streamlit as st
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Save kaggle.json from Streamlit secrets
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kaggle_json_path, "w") as fp:
        fp.write(st.secrets["KAGGLE_JSON"])
    os.chmod(kaggle_json_path, 0o600)

    # Download files
    api = KaggleApi()
    api.authenticate()
    st.info("Downloading Kaggle dataset artifacts if not already present...")
    api.dataset_download_files(
        "sonalikasingh17/massive-pickle-files",
        path="artifacts",
        unzip=True
    )
    st.success("Kaggle artifacts downloaded and ready.")


    def ensure_artifacts_exist():
    # Only download if needed
        important_files = [
            "language_pipeline.pkl",
            "continent_lda_model.pkl",
            "continent_qda_model.pkl",
            "continent_vectorizer.pkl",
            "continent_svd.pkl",
            "continent_label_encoder.pkl",
        ]
        missing = [f for f in important_files if not os.path.exists(os.path.join("artifacts", f))]
        if missing:
        download_from_kaggle()



# def ensure_artifacts_exist():
#     """Download all .pkl files from the Kaggle dataset once."""
#     KAGGLE_BASE = (
#         "https://www.kaggle.com/datasets/"
#         "sonalikasingh17/massive-pickle-files/download?file="
#     )
#     artifact_files = [
#         # language pipeline  (for fast inference)
#         "language_pipeline.pkl",
#         # continent models
#         "continent_lda_model.pkl",
#         "continent_qda_model.pkl",
#         # vectorisers / reducers
#         "continent_vectorizer.pkl",
#         "continent_svd.pkl",
#         # label encoders
#         "continent_label_encoder.pkl",
#         "label_encoder.pkl",
#         # optional extras
#         "language_model.pkl",
#         "language_vectorizer.pkl",
#         "model_performance.pkl"
#     ]

#     os.makedirs("artifacts", exist_ok=True)

#     for fname in artifact_files:
#         path = os.path.join("artifacts", fname)
#         if os.path.exists(path):
#             continue                                 

#         # user feedback (works both in Streamlit & CLI)
#         try:
#             st.info(f"Downloading {fname} from Kaggle…")
#         except Exception:
#             print(f"[INFO] Downloading {fname}")

#         url = KAGGLE_BASE + fname
#         with requests.get(url, stream=True) as r:
#             r.raise_for_status()
#             with open(path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)

#         try:
#             st.success(f"Downloaded {fname}")
#         except Exception:
#             print(f"[INFO] Downloaded {fname}")



def save_object(file_path: str, obj: Any) -> None:
    """Save object to file using joblib"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

        logger.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logger.error(f"Error saving object: {str(e)}")
        raise CustomException(e, sys)

def load_object(file_path: str) -> Any:
    """Load object from file using joblib"""

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            print("DEBUG: File exists?", os.path.exists(file_path), "Path:", file_path)
            print("DEBUG: File size:", os.path.getsize(file_path))
            obj = joblib.load(file_obj)

        logger.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logger.error(f"Error loading object: {str(e)}")
        raise CustomException(e, sys)

def evaluate_model(y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
    """Evaluate model performance"""
    try:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']
        weighted_f1 = report['weighted avg']['f1-score']

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1_score': weighted_f1,
            'classification_report': report
        }

        logger.info(f"Model evaluation completed for {model_name}")
        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise CustomException(e, sys)

def get_language_continent_mapping():
    """Get mapping of language locales to continents"""
    continent_lookup = {
        'ZA': 'Africa', 'KE': 'Africa', 'AL': 'Europe', 'GB': 'Europe', 'DK': 'Europe', 'DE': 'Europe',
        'ES': 'Europe', 'FR': 'Europe', 'FI': 'Europe', 'HU': 'Europe', 'IS': 'Europe', 'IT': 'Europe',
        'ID': 'Asia', 'LV': 'Europe', 'MY': 'Asia', 'NO': 'Europe', 'NL': 'Europe', 'PL': 'Europe',
        'PT': 'Europe', 'RO': 'Europe', 'RU': 'Europe', 'SL': 'Europe', 'SE': 'Europe', 'PH': 'Asia',
        'TR': 'Asia', 'VN': 'Asia', 'US': 'North America'
    }

    def map_continent(locale):
        country = locale.split('-')[1]
        return continent_lookup.get(country, 'Unknown')

    return map_continent

def get_supported_languages() -> List[str]:
    """Get list of supported languages"""
    return [
        'af-ZA', 'da-DK', 'de-DE', 'en-US', 'es-ES', 'fr-FR', 'fi-FI', 'hu-HU', 'is-IS', 'it-IT',
        'jv-ID', 'lv-LV', 'ms-MY', 'nb-NO', 'nl-NL', 'pl-PL', 'pt-PT', 'ro-RO', 'ru-RU', 'sl-SL',
        'sv-SE', 'sq-AL', 'sw-KE', 'tl-PH', 'tr-TR', 'vi-VN', 'cy-GB'
    ]

def create_directory(directory_path: str) -> None:
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory created/verified: {directory_path}")
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")
        raise CustomException(e, sys)
