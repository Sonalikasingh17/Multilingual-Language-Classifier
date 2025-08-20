import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from datasets import load_dataset
from typing import Tuple, Dict, List

from src.exception import CustomException, DataIngestionError
from src.logger import get_logger
from src.utils import create_directory, save_object, get_supported_languages, get_language_continent_mapping

logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component"""
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    validation_data_path: str = os.path.join('artifacts', "validation.csv")
    raw_data_path: str = os.path.join('artifacts', "raw_data.csv")
    processed_data_path: str = os.path.join('artifacts', "processed_data.csv")

class DataIngestion:
    """
    Data Ingestion component for loading and preparing the MASSIVE dataset
    """
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def load_massive_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load MASSIVE dataset for all supported languages
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, validation, test dataframes
        """
        try:
            logger.info("Starting MASSIVE dataset loading")
            
            languages = get_supported_languages()
            logger.info(f"Loading data for {len(languages)} languages: {languages}")
            
            # Initialize lists to store data
            all_train_data = []
            all_val_data = []
            all_test_data = []
            
            # Load data for each language
            for i, lang in enumerate(languages):
                try:
                    logger.info(f"Loading data for language {i+1}/{len(languages)}: {lang}")
                    
                    # Load train split
                    train_dataset = load_dataset("qanastek/MASSIVE", lang, split='train', trust_remote_code=True)
                    train_df = pd.DataFrame(train_dataset)
                    train_df = train_df[['locale', 'utt']].copy()
                    train_df['split'] = 'train'
                    all_train_data.append(train_df)
                    
                    # Load validation split
                    val_dataset = load_dataset("qanastek/MASSIVE", lang, split='validation', trust_remote_code=True)
                    val_df = pd.DataFrame(val_dataset)
                    val_df = val_df[['locale', 'utt']].copy()
                    val_df['split'] = 'validation'
                    all_val_data.append(val_df)
                    
                    # Load test split
                    test_dataset = load_dataset("qanastek/MASSIVE", lang, split='test', trust_remote_code=True)
                    test_df = pd.DataFrame(test_dataset)
                    test_df = test_df[['locale', 'utt']].copy()
                    test_df['split'] = 'test'
                    all_test_data.append(test_df)
                    
                    logger.info(f"Successfully loaded {lang}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load data for language {lang}: {str(e)}")
                    continue
            
            # Concatenate all data
            train_data = pd.concat(all_train_data, ignore_index=True)
            val_data = pd.concat(all_val_data, ignore_index=True)
            test_data = pd.concat(all_test_data, ignore_index=True)
            
            logger.info(f"Dataset loading completed successfully")
            logger.info(f"Train data shape: {train_data.shape}")
            logger.info(f"Validation data shape: {val_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error in loading MASSIVE dataset: {str(e)}")
            raise DataIngestionError(e, sys)
    
    def add_continent_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add continent labels to the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe with locale column
            
        Returns:
            pd.DataFrame: Dataframe with continent column added
        """
        try:
            map_continent = get_language_continent_mapping()
            df['continent'] = df['locale'].apply(map_continent)
            
            logger.info("Continent labels added successfully")
            logger.info(f"Continent distribution: {df['continent'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding continent labels: {str(e)}")
            raise DataIngestionError(e, sys)
    
    def save_data_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Save data splits to CSV files
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data  
            test_df (pd.DataFrame): Test data
        """
        try:
            # Create artifacts directory
            create_directory('artifacts')
            
            # Save individual splits
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            val_df.to_csv(self.ingestion_config.validation_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            
            # Save combined raw data
            all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
            all_data.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logger.info("Data splits saved successfully")
            logger.info(f"Train data saved to: {self.ingestion_config.train_data_path}")
            logger.info(f"Validation data saved to: {self.ingestion_config.validation_data_path}")
            logger.info(f"Test data saved to: {self.ingestion_config.test_data_path}")
            logger.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")
            
        except Exception as e:
            logger.error(f"Error saving data splits: {str(e)}")
            raise DataIngestionError(e, sys)
    
    def get_data_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Get statistics about the loaded data
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            test_df (pd.DataFrame): Test data
            
        Returns:
            Dict: Dictionary containing data statistics
        """
        try:
            stats = {
                'total_samples': len(train_df) + len(val_df) + len(test_df),
                'train_samples': len(train_df),
                'validation_samples': len(val_df),
                'test_samples': len(test_df),
                'languages': train_df['locale'].nunique(),
                'language_list': sorted(train_df['locale'].unique().tolist()),
                'continents': train_df['continent'].nunique() if 'continent' in train_df.columns else 0,
                'continent_list': sorted(train_df['continent'].unique().tolist()) if 'continent' in train_df.columns else [],
                'avg_utterance_length': train_df['utt'].str.len().mean(),
                'min_utterance_length': train_df['utt'].str.len().min(),
                'max_utterance_length': train_df['utt'].str.len().max()
            }
            
            logger.info("Data statistics computed successfully")
            for key, value in stats.items():
                if isinstance(value, (list,)) and len(value) > 10:
                    logger.info(f"{key}: {len(value)} items")
                else:
                    logger.info(f"{key}: {value}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing data statistics: {str(e)}")
            raise DataIngestionError(e, sys)
    
    def initiate_data_ingestion(self) -> Tuple[str, str, str]:
        """
        Initiate the complete data ingestion process
        
        Returns:
            Tuple[str, str, str]: Paths to train, validation, and test data files
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING DATA INGESTION PROCESS")
            logger.info("=" * 50)
            
            # Load dataset
            train_df, val_df, test_df = self.load_massive_dataset()
            
            # Add continent labels
            train_df = self.add_continent_labels(train_df)
            val_df = self.add_continent_labels(val_df)
            test_df = self.add_continent_labels(test_df)
            
            # Get and log statistics
            stats = self.get_data_statistics(train_df, val_df, test_df)
            
            # Save data splits
            self.save_data_splits(train_df, val_df, test_df)
            
            logger.info("=" * 50)
            logger.info("DATA INGESTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logger.error(f"Error in data ingestion process: {str(e)}")
            raise DataIngestionError(e, sys)

if __name__ == "__main__":
    # Test data ingestion
    data_ingestion = DataIngestion()
    train_path, val_path, test_path = data_ingestion.initiate_data_ingestion()
    print(f"Data ingestion completed. Files saved at:")
    print(f"Train: {train_path}")
    print(f"Validation: {val_path}")
    print(f"Test: {test_path}")
