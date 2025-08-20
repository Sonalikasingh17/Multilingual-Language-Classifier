import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Dict, Any

from src.exception import CustomException, ModelTrainingError
from src.logger import get_logger, log_model_performance
from src.utils import save_object, evaluate_model, create_directory

logger = get_logger(__name__)

@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer component"""
    language_model_file_path: str = os.path.join("artifacts", "language_model.pkl")
    continent_lda_model_file_path: str = os.path.join("artifacts", "continent_lda_model.pkl")
    continent_qda_model_file_path: str = os.path.join("artifacts", "continent_qda_model.pkl")
    language_pipeline_path: str = os.path.join("artifacts", "language_pipeline.pkl")
    model_performance_path: str = os.path.join("artifacts", "model_performance.pkl")

class ModelTrainer:
    """
    Model Trainer component for training machine learning models
    """
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def create_language_model(self) -> MultinomialNB:
        """
        Create Multinomial Naive Bayes model for language classification
        
        Returns:
            MultinomialNB: Configured Naive Bayes model
        """
        try:
            logger.info("Creating Multinomial Naive Bayes model for language classification")
            
            model = MultinomialNB(
                alpha=0.1,  # Laplace smoothing
                fit_prior=True
            )
            
            logger.info("Language model created successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error creating language model: {str(e)}")
            raise ModelTrainingError(e, sys)
    
    def create_continent_models(self) -> Tuple[LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis]:
        """
        Create LDA and QDA models for continent classification
        
        Returns:
            Tuple[LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis]: LDA and QDA models
        """
        try:
            logger.info("Creating LDA and QDA models for continent classification")
            
            # Linear Discriminant Analysis
            lda_model = LinearDiscriminantAnalysis(
                solver='svd',  # SVD solver for stability
                shrinkage=None
            )
            
            # Quadratic Discriminant Analysis
            qda_model = QuadraticDiscriminantAnalysis(
                reg_param=0.1  # Regularization for stability
            )
            
            logger.info("Continent models created successfully")
            return lda_model, qda_model
            
        except Exception as e:
            logger.error(f"Error creating continent models: {str(e)}")
            raise ModelTrainingError(e, sys)
    
    def train_language_model(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict[str, Any]:
        """
        Train and evaluate language classification model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Dict[str, Any]: Model performance metrics
        """
        try:
            logger.info("=" * 40)
            logger.info("TRAINING LANGUAGE CLASSIFICATION MODEL")
            logger.info("=" * 40)
            
            # Create and train model
            model = self.create_language_model()
            
            logger.info("Training Multinomial Naive Bayes model...")
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            # Calculate performance metrics
            train_metrics = evaluate_model(y_train, train_pred, "Language_Train")
            val_metrics = evaluate_model(y_val, val_pred, "Language_Validation")
            test_metrics = evaluate_model(y_test, test_pred, "Language_Test")
            
            # Log performance
            log_model_performance(train_metrics, "Language Classification - Training", logger)
            log_model_performance(val_metrics, "Language Classification - Validation", logger)
            log_model_performance(test_metrics, "Language Classification - Test", logger)
            
            # Save model
            save_object(self.model_trainer_config.language_model_file_path, model)
            
            # Prepare performance summary
            performance = {
                'model_type': 'Multinomial Naive Bayes',
                'task': 'Language Classification',
                'train_accuracy': train_metrics['accuracy'],
                'validation_accuracy': val_metrics['accuracy'], 
                'test_accuracy': test_metrics['accuracy'],
                'train_f1_score': train_metrics['weighted_f1_score'],
                'validation_f1_score': val_metrics['weighted_f1_score'],
                'test_f1_score': test_metrics['weighted_f1_score'],
                'n_classes': len(np.unique(y_train)),
                'n_train_samples': len(y_train),
                'n_val_samples': len(y_val),
                'n_test_samples': len(y_test)
            }
            
            logger.info("Language model training completed successfully")
            return performance
            
        except Exception as e:
            logger.error(f"Error training language model: {str(e)}")
            raise ModelTrainingError(e, sys)
    
    def train_continent_models(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict[str, Any]:
        """
        Train and evaluate continent classification models (LDA and QDA)
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Dict[str, Any]: Model performance metrics for both LDA and QDA
        """
        try:
            logger.info("=" * 40)
            logger.info("TRAINING CONTINENT CLASSIFICATION MODELS")
            logger.info("=" * 40)
            
            # Create models
            lda_model, qda_model = self.create_continent_models()
            
            # Train LDA model
            logger.info("Training Linear Discriminant Analysis (LDA) model...")
            lda_model.fit(X_train, y_train)
            
            # LDA predictions
            lda_train_pred = lda_model.predict(X_train)
            lda_val_pred = lda_model.predict(X_val)
            lda_test_pred = lda_model.predict(X_test)
            
            # LDA performance metrics
            lda_train_metrics = evaluate_model(y_train, lda_train_pred, "Continent_LDA_Train")
            lda_val_metrics = evaluate_model(y_val, lda_val_pred, "Continent_LDA_Validation")
            lda_test_metrics = evaluate_model(y_test, lda_test_pred, "Continent_LDA_Test")
            
            # Log LDA performance
            log_model_performance(lda_train_metrics, "Continent LDA - Training", logger)
            log_model_performance(lda_val_metrics, "Continent LDA - Validation", logger)
            log_model_performance(lda_test_metrics, "Continent LDA - Test", logger)
            
            # Train QDA model
            logger.info("Training Quadratic Discriminant Analysis (QDA) model...")
            qda_model.fit(X_train, y_train)
            
            # QDA predictions
            qda_train_pred = qda_model.predict(X_train)
            qda_val_pred = qda_model.predict(X_val)
            qda_test_pred = qda_model.predict(X_test)
            
            # QDA performance metrics
            qda_train_metrics = evaluate_model(y_train, qda_train_pred, "Continent_QDA_Train")
            qda_val_metrics = evaluate_model(y_val, qda_val_pred, "Continent_QDA_Validation")
            qda_test_metrics = evaluate_model(y_test, qda_test_pred, "Continent_QDA_Test")
            
            # Log QDA performance
            log_model_performance(qda_train_metrics, "Continent QDA - Training", logger)
            log_model_performance(qda_val_metrics, "Continent QDA - Validation", logger)
            log_model_performance(qda_test_metrics, "Continent QDA - Test", logger)
            
            # Save models
            save_object(self.model_trainer_config.continent_lda_model_file_path, lda_model)
            save_object(self.model_trainer_config.continent_qda_model_file_path, qda_model)
            
            # Prepare performance summary
            performance = {
                'lda': {
                    'model_type': 'Linear Discriminant Analysis',
                    'task': 'Continent Classification',
                    'train_accuracy': lda_train_metrics['accuracy'],
                    'validation_accuracy': lda_val_metrics['accuracy'],
                    'test_accuracy': lda_test_metrics['accuracy'],
                    'train_f1_score': lda_train_metrics['weighted_f1_score'],
                    'validation_f1_score': lda_val_metrics['weighted_f1_score'],
                    'test_f1_score': lda_test_metrics['weighted_f1_score'],
                },
                'qda': {
                    'model_type': 'Quadratic Discriminant Analysis',
                    'task': 'Continent Classification',
                    'train_accuracy': qda_train_metrics['accuracy'],
                    'validation_accuracy': qda_val_metrics['accuracy'],
                    'test_accuracy': qda_test_metrics['accuracy'],
                    'train_f1_score': qda_train_metrics['weighted_f1_score'],
                    'validation_f1_score': qda_val_metrics['weighted_f1_score'],
                    'test_f1_score': qda_test_metrics['weighted_f1_score'],
                },
                'best_model': 'lda' if lda_val_metrics['accuracy'] > qda_val_metrics['accuracy'] else 'qda',
                'n_classes': len(np.unique(y_train)),
                'n_train_samples': len(y_train),
                'n_val_samples': len(y_val),
                'n_test_samples': len(y_test)
            }
            
            logger.info("Continent models training completed successfully")
            logger.info(f"Best performing model: {performance['best_model'].upper()}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Error training continent models: {str(e)}")
            raise ModelTrainingError(e, sys)
    
    def create_language_pipeline(self) -> Pipeline:
        """
        Create end-to-end pipeline for language classification
        
        Returns:
            Pipeline: Complete pipeline for language classification
        """
        try:
            logger.info("Creating language classification pipeline")
            
            # Create pipeline with vectorizer and model
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.8,
                    lowercase=True,
                    strip_accents='unicode',
                    analyzer='char_wb'
                )),
                ('nb', MultinomialNB(alpha=0.1, fit_prior=True))
            ])
            
            logger.info("Language pipeline created successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error creating language pipeline: {str(e)}")
            raise ModelTrainingError(e, sys)
    
    def train_complete_language_pipeline(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train complete end-to-end language classification pipeline
        
        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data
            test_df (pd.DataFrame): Test data
            
        Returns:
            Dict[str, Any]: Pipeline performance metrics
        """
        try:
            logger.info("Training complete language classification pipeline")
            
            # Create pipeline
            pipeline = self.create_language_pipeline()
            
            # Prepare data
            X_train = train_df['utt']
            y_train = train_df['locale']
            X_val = val_df['utt']
            y_val = val_df['locale']
            X_test = test_df['utt']
            y_test = test_df['locale']
            
            # Train pipeline
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            train_pred = pipeline.predict(X_train)
            val_pred = pipeline.predict(X_val)
            test_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            val_accuracy = accuracy_score(y_val, val_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Save pipeline
            save_object(self.model_trainer_config.language_pipeline_path, pipeline)
            
            performance = {
                'model_type': 'TF-IDF + Multinomial Naive Bayes Pipeline',
                'task': 'Language Classification',
                'train_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy,
                'test_accuracy': test_accuracy
            }
            
            logger.info("Language pipeline training completed successfully")
            log_model_performance(performance, "Language Pipeline", logger)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error training language pipeline: {str(e)}")
            raise ModelTrainingError(e, sys)
    
    def initiate_model_trainer(self, transformed_data: Dict[str, Dict[str, Tuple]]) -> Dict[str, Any]:
        """
        Initiate the complete model training process
        
        Args:
            transformed_data (Dict): Transformed data from data transformation component
            
        Returns:
            Dict[str, Any]: Complete model performance metrics
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL TRAINING PROCESS")
            logger.info("=" * 60)
            
            # Create artifacts directory
            create_directory('artifacts')
            
            # Extract language classification data
            lang_train = transformed_data['language']['train']
            lang_val = transformed_data['language']['validation']
            lang_test = transformed_data['language']['test']
            
            # Extract continent classification data
            cont_train = transformed_data['continent']['train']
            cont_val = transformed_data['continent']['validation']
            cont_test = transformed_data['continent']['test']
            
            # Train language classification model
            language_performance = self.train_language_model(
                lang_train[0], lang_train[1],
                lang_val[0], lang_val[1],
                lang_test[0], lang_test[1]
            )
            
            # Train continent classification models
            continent_performance = self.train_continent_models(
                cont_train[0], cont_train[1],
                cont_val[0], cont_val[1],
                cont_test[0], cont_test[1]
            )
            
            # Combine all performance metrics
            all_performance = {
                'language_classification': language_performance,
                'continent_classification': continent_performance,
                'summary': {
                    'language_best_accuracy': language_performance['test_accuracy'],
                    'continent_lda_accuracy': continent_performance['lda']['test_accuracy'],
                    'continent_qda_accuracy': continent_performance['qda']['test_accuracy'],
                    'continent_best_accuracy': max(
                        continent_performance['lda']['test_accuracy'],
                        continent_performance['qda']['test_accuracy']
                    )
                }
            }
            
            # Save performance metrics
            save_object(self.model_trainer_config.model_performance_path, all_performance)
            
            logger.info("=" * 60)
            logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info("FINAL PERFORMANCE SUMMARY:")
            logger.info(f"Language Classification Test Accuracy: {all_performance['summary']['language_best_accuracy']:.4f}")
            logger.info(f"Continent LDA Test Accuracy: {all_performance['summary']['continent_lda_accuracy']:.4f}")
            logger.info(f"Continent QDA Test Accuracy: {all_performance['summary']['continent_qda_accuracy']:.4f}")
            logger.info("=" * 60)
            
            return all_performance
            
        except Exception as e:
            logger.error(f"Error in model training process: {str(e)}")
            raise ModelTrainingError(e, sys)

if __name__ == "__main__":
    # Test model training
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    # Get and transform data
    data_ingestion = DataIngestion()
    train_path, val_path, test_path = data_ingestion.initiate_data_ingestion()
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    data_transformation = DataTransformation()
    transformed_data = data_transformation.transform_data(train_df, val_df, test_df)
    
    # Train models
    model_trainer = ModelTrainer()
    performance = model_trainer.initiate_model_trainer(transformed_data)
    
    print("Model training completed successfully!")
    print(f"Language Classification Accuracy: {performance['summary']['language_best_accuracy']:.4f}")
    print(f"Best Continent Classification Accuracy: {performance['summary']['continent_best_accuracy']:.4f}")
