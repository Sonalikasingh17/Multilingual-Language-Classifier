import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any

from src.exception import CustomException, DataTransformationError
from src.logger import get_logger
from src.utils import save_object, create_directory

logger = get_logger(__name__)

@dataclass 
class DataTransformationConfig:
    """Configuration for data transformation component"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")
    language_vectorizer_path: str = os.path.join('artifacts', "language_vectorizer.pkl")
    continent_vectorizer_path: str = os.path.join('artifacts', "continent_vectorizer.pkl")
    continent_svd_path: str = os.path.join('artifacts', "continent_svd.pkl")
    label_encoder_path: str = os.path.join('artifacts', "label_encoder.pkl")
    continent_label_encoder_path: str = os.path.join('artifacts', "continent_label_encoder.pkl")

class DataTransformation:
    """
    Data Transformation component for preprocessing text data and creating features
    """
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def create_language_vectorizer(self) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer for language classification
        
        Returns:
            TfidfVectorizer: Configured vectorizer for language classification
        """
        try:
            logger.info("Creating TF-IDF vectorizer for language classification")
            
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8,
                lowercase=True,
                strip_accents='unicode',
                analyzer='char_wb'  # Character-level n-grams work well for language detection
            )
            
            logger.info("Language vectorizer created successfully")
            return vectorizer
            
        except Exception as e:
            logger.error(f"Error creating language vectorizer: {str(e)}")
            raise DataTransformationError(e, sys)
    
    def create_continent_vectorizer(self) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer for continent classification
        
        Returns:
            TfidfVectorizer: Configured vectorizer for continent classification
        """
        try:
            logger.info("Creating TF-IDF vectorizer for continent classification")
            
            vectorizer = TfidfVectorizer(
                max_features=15000,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.7,
                lowercase=True,
                strip_accents='unicode',
                analyzer='word'  # Word-level features for content understanding
            )
            
            logger.info("Continent vectorizer created successfully")
            return vectorizer
            
        except Exception as e:
            logger.error(f"Error creating continent vectorizer: {str(e)}")
            raise DataTransformationError(e, sys)
    
    def create_svd_reducer(self, n_components: int = 100) -> TruncatedSVD:
        """
        Create SVD for dimensionality reduction
        
        Args:
            n_components (int): Number of components for SVD
            
        Returns:
            TruncatedSVD: Configured SVD reducer
        """
        try:
            logger.info(f"Creating SVD reducer with {n_components} components")
            
            svd = TruncatedSVD(
                n_components=n_components,
                random_state=42
            )
            
            logger.info("SVD reducer created successfully")
            return svd
            
        except Exception as e:
            logger.error(f"Error creating SVD reducer: {str(e)}")
            raise DataTransformationError(e, sys)
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        try:
            if pd.isna(text) or text is None:
                return ""
            
            # Convert to string and basic cleaning
            text = str(text)
            text = text.strip()
            
            # Remove extra whitespaces
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return str(text) if text else ""
    
    def prepare_language_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for language classification
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels for language classification
        """
        try:
            logger.info("Preparing data for language classification")
            
            # Preprocess text
            df['processed_utt'] = df['utt'].apply(self.preprocess_text)
            
            # Create vectorizer and transform text
            vectorizer = self.create_language_vectorizer()
            X = vectorizer.fit_transform(df['processed_utt'])
            
            # Encode labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df['locale'])
            
            # Save vectorizer and label encoder
            save_object(self.data_transformation_config.language_vectorizer_path, vectorizer)
            save_object(self.data_transformation_config.label_encoder_path, label_encoder)
            
            logger.info(f"Language data prepared - Features shape: {X.shape}, Labels shape: {y.shape}")
            logger.info(f"Number of unique languages: {len(label_encoder.classes_)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing language data: {str(e)}")
            raise DataTransformationError(e, sys)
    
    def prepare_continent_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for continent classification
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels for continent classification
        """
        try:
            logger.info("Preparing data for continent classification")
            
            # Preprocess text
            df['processed_utt'] = df['utt'].apply(self.preprocess_text)
            
            # Create vectorizer and transform text
            vectorizer = self.create_continent_vectorizer()
            X_tfidf = vectorizer.fit_transform(df['processed_utt'])
            
            # Apply SVD for dimensionality reduction
            svd = self.create_svd_reducer()
            X = svd.fit_transform(X_tfidf)
            
            # Encode labels
            continent_label_encoder = LabelEncoder()
            y = continent_label_encoder.fit_transform(df['continent'])
            
            # Save vectorizer, SVD, and label encoder
            save_object(self.data_transformation_config.continent_vectorizer_path, vectorizer)
            save_object(self.data_transformation_config.continent_svd_path, svd)
            save_object(self.data_transformation_config.continent_label_encoder_path, continent_label_encoder)
            
            logger.info(f"Continent data prepared - Features shape: {X.shape}, Labels shape: {y.shape}")
            logger.info(f"Number of unique continents: {len(continent_label_encoder.classes_)}")
            logger.info(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing continent data: {str(e)}")
            raise DataTransformationError(e, sys)
    
    def get_feature_statistics(self, X: np.ndarray, task_name: str) -> Dict[str, Any]:
        """
        Get statistics about the feature matrix
        
        Args:
            X (np.ndarray): Feature matrix
            task_name (str): Name of the task
            
        Returns:
            Dict[str, Any]: Statistics about features
        """
        try:
            stats = {
                'task': task_name,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'sparsity': 1.0 - (np.count_nonzero(X) / X.size) if hasattr(X, 'size') else 'N/A',
                'mean_feature_value': np.mean(X) if hasattr(X, 'mean') else 'N/A',
                'std_feature_value': np.std(X) if hasattr(X, 'std') else 'N/A'
            }
            
            logger.info(f"Feature statistics for {task_name}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing feature statistics: {str(e)}")
            return {'error': str(e)}
    
    def transform_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform all datasets for both language and continent classification
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            test_df (pd.DataFrame): Test data
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Transformed data for both tasks
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING DATA TRANSFORMATION")
            logger.info("=" * 50)
            
            # Create artifacts directory
            create_directory('artifacts')
            
            # Combine all data for fitting transformers
            all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
            
            # Prepare language classification data
            logger.info("Preparing language classification features...")
            X_lang, y_lang = self.prepare_language_data(all_data.copy())
            
            # Prepare continent classification data  
            logger.info("Preparing continent classification features...")
            X_cont, y_cont = self.prepare_continent_data(all_data.copy())
            
            # Split back into train/val/test
            train_size = len(train_df)
            val_size = len(val_df)
            
            # Language classification splits
            X_lang_train = X_lang[:train_size]
            X_lang_val = X_lang[train_size:train_size + val_size]
            X_lang_test = X_lang[train_size + val_size:]
            
            y_lang_train = y_lang[:train_size]
            y_lang_val = y_lang[train_size:train_size + val_size]
            y_lang_test = y_lang[train_size + val_size:]
            
            # Continent classification splits
            X_cont_train = X_cont[:train_size]
            X_cont_val = X_cont[train_size:train_size + val_size]
            X_cont_test = X_cont[train_size + val_size:]
            
            y_cont_train = y_cont[:train_size]
            y_cont_val = y_cont[train_size:train_size + val_size]
            y_cont_test = y_cont[train_size + val_size:]
            
            # Get feature statistics
            self.get_feature_statistics(X_lang_train, "Language Classification")
            self.get_feature_statistics(X_cont_train, "Continent Classification")
            
            # Prepare return data
            transformed_data = {
                'language': {
                    'train': (X_lang_train, y_lang_train),
                    'validation': (X_lang_val, y_lang_val),
                    'test': (X_lang_test, y_lang_test)
                },
                'continent': {
                    'train': (X_cont_train, y_cont_train),
                    'validation': (X_cont_val, y_cont_val),
                    'test': (X_cont_test, y_cont_test)
                }
            }
            
            logger.info("=" * 50)
            logger.info("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise DataTransformationError(e, sys)

if __name__ == "__main__":
    # Test data transformation
    from src.components.data_ingestion import DataIngestion
    
    # First get the data
    data_ingestion = DataIngestion()
    train_path, val_path, test_path = data_ingestion.initiate_data_ingestion()
    
    # Load the data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # Transform the data
    data_transformation = DataTransformation()
    transformed_data = data_transformation.transform_data(train_df, val_df, test_df)
    
    print("Data transformation completed successfully!")
    print(f"Language classification - Train: {transformed_data['language']['train'][0].shape}")
    print(f"Continent classification - Train: {transformed_data['continent']['train'][0].shape}")
