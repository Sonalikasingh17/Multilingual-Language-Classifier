import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from src.exception import CustomException, ModelPredictionError
from src.logger import get_logger
from src.utils import load_object, get_supported_languages, get_language_continent_mapping

logger = get_logger(__name__)

@dataclass
class PredictPipelineConfig:
    """Configuration for prediction pipeline"""
    language_model_path: str = os.path.join("artifacts", "language_model.pkl")
    continent_lda_model_path: str = os.path.join("artifacts", "continent_lda_model.pkl")
    continent_qda_model_path: str = os.path.join("artifacts", "continent_qda_model.pkl")
    language_pipeline_path: str = os.path.join("artifacts", "language_pipeline.pkl")
    language_vectorizer_path: str = os.path.join("artifacts", "language_vectorizer.pkl")
    continent_vectorizer_path: str = os.path.join("artifacts", "continent_vectorizer.pkl")
    continent_svd_path: str = os.path.join("artifacts", "continent_svd.pkl")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")
    continent_label_encoder_path: str = os.path.join("artifacts", "continent_label_encoder.pkl")

class PredictionPipeline:
    """
    Prediction pipeline for multilingual language and continent classification
    """
    
    def __init__(self):
        self.config = PredictPipelineConfig()
        self._models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and preprocessors"""
        try:
            logger.info("Loading trained models and preprocessors...")
            
            # Check if all required files exist
            required_files = [
                self.config.language_pipeline_path,
                self.config.continent_lda_model_path,
                self.config.continent_qda_model_path,
                self.config.continent_vectorizer_path,
                self.config.continent_svd_path,
                self.config.continent_label_encoder_path
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {missing_files}")
            
            # Load language classification pipeline
            self.language_pipeline = load_object(self.config.language_pipeline_path)
            
            # Load continent classification components
            self.continent_lda_model = load_object(self.config.continent_lda_model_path)
            self.continent_qda_model = load_object(self.config.continent_qda_model_path)
            self.continent_vectorizer = load_object(self.config.continent_vectorizer_path)
            self.continent_svd = load_object(self.config.continent_svd_path)
            self.continent_label_encoder = load_object(self.config.continent_label_encoder_path)
            
            # Get helper functions
            self.map_continent = get_language_continent_mapping()
            self.supported_languages = get_supported_languages()
            
            self._models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise ModelPredictionError(e, sys)
    
    def predict_language(self, text: str) -> Dict[str, Any]:
        """
        Predict the language of input text
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Language prediction results
        """
        try:
            if not self._models_loaded:
                raise RuntimeError("Models not loaded. Please ensure training has been completed.")
            
            # Clean input text
            if not text or text.strip() == "":
                raise ValueError("Input text cannot be empty")
            
            text = text.strip()
            
            # Make prediction using the pipeline
            predicted_language = self.language_pipeline.predict([text])[0]
            
            # Get prediction probabilities
            try:
                # Get probabilities from the pipeline
                probabilities = self.language_pipeline.predict_proba([text])[0]
                classes = self.language_pipeline.classes_
                
                # Create probability dictionary
                prob_dict = dict(zip(classes, probabilities))
                
                # Get top 5 predictions
                top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                
                confidence = prob_dict[predicted_language]
                
            except Exception as prob_error:
                logger.warning(f"Could not get probabilities: {prob_error}")
                confidence = None
                top_predictions = [(predicted_language, None)]
            
            # Get continent from predicted language
            predicted_continent = self.map_continent(predicted_language)
            
            result = {
                'predicted_language': predicted_language,
                'predicted_continent': predicted_continent,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'input_text': text,
                'text_length': len(text),
                'is_supported': predicted_language in self.supported_languages
            }
            
            if confidence is not None:
                logger.info(f"Language prediction completed: {predicted_language} (confidence: {confidence:.4f})")
            else:
                logger.info(f"Language prediction completed: {predicted_language} (confidence: N/A)")

            return result
            
        except Exception as e:
            logger.error(f"Error in language prediction: {str(e)}")
            raise ModelPredictionError(e, sys)
    
    def predict_continent(self, text: str, model_type: str = 'lda') -> Dict[str, Any]:
        """
        Predict the continent based on input text
        
        Args:
            text (str): Input text
            model_type (str): Type of model to use ('lda' or 'qda')
            
        Returns:
            Dict[str, Any]: Continent prediction results
        """
        try:
            if not self._models_loaded:
                raise RuntimeError("Models not loaded. Please ensure training has been completed.")
            
            # Clean input text
            if not text or text.strip() == "":
                raise ValueError("Input text cannot be empty")
            
            text = text.strip()
            
            # Validate model type
            if model_type not in ['lda', 'qda']:
                raise ValueError("model_type must be 'lda' or 'qda'")
            
            # Preprocess text using the same pipeline as training
            text_vectorized = self.continent_vectorizer.transform([text])
            text_reduced = self.continent_svd.transform(text_vectorized)
            
            # Select model
            model = self.continent_lda_model if model_type == 'lda' else self.continent_qda_model
            
            # Make prediction
            predicted_continent_encoded = model.predict(text_reduced)[0]
            predicted_continent = self.continent_label_encoder.inverse_transform([predicted_continent_encoded])[0]
            
            # Get prediction probabilities
            try:
                probabilities = model.predict_proba(text_reduced)[0]
                classes = self.continent_label_encoder.inverse_transform(model.classes_)
                
                prob_dict = dict(zip(classes, probabilities))
                confidence = prob_dict[predicted_continent]
                
                # Get top predictions
                top_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                
            except Exception as prob_error:
                logger.warning(f"Could not get probabilities: {prob_error}")
                confidence = None
                top_predictions = [(predicted_continent, None)]
            
            result = {
                'predicted_continent': predicted_continent,
                'model_used': model_type.upper(),
                'confidence': confidence,
                'top_predictions': top_predictions,
                'input_text': text,
                'text_length': len(text)
            }

            if confidence is not None:
                logger.info(f"Continent prediction completed: {predicted_continent} using {model_type.upper()} (confidence: {confidence:.4f})")
            else:
                logger.info(f"Continent prediction completed: {predicted_continent} using {model_type.upper()} (confidence: N/A)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in continent prediction: {str(e)}")
            raise ModelPredictionError(e, sys)
    
    def predict_both(self, text: str, continent_model: str = 'lda') -> Dict[str, Any]:
        """
        Predict both language and continent for input text
        
        Args:
            text (str): Input text
            continent_model (str): Model to use for continent prediction ('lda' or 'qda')
            
        Returns:
            Dict[str, Any]: Combined prediction results
        """
        try:
            # Get both predictions
            language_result = self.predict_language(text)
            continent_result = self.predict_continent(text, continent_model)
            
            # Combine results
            combined_result = {
                'input_text': text,
                'text_length': len(text),
                'language_prediction': {
                    'predicted_language': language_result['predicted_language'],
                    'confidence': language_result['confidence'],
                    'top_predictions': language_result['top_predictions'][:3],  # Top 3
                    'continent_from_language': language_result['predicted_continent']
                },
                'continent_prediction': {
                    'predicted_continent': continent_result['predicted_continent'],
                    'model_used': continent_result['model_used'],
                    'confidence': continent_result['confidence'],
                    'top_predictions': continent_result['top_predictions']
                },
                'consistency_check': {
                    'language_continent': language_result['predicted_continent'],
                    'direct_continent': continent_result['predicted_continent'],
                    'consistent': language_result['predicted_continent'] == continent_result['predicted_continent']
                }
            }
            
            logger.info(f"Combined prediction completed for text: '{text[:50]}...'")
            logger.info(f"Language: {language_result['predicted_language']}, Continent: {continent_result['predicted_continent']}")
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error in combined prediction: {str(e)}")
            raise ModelPredictionError(e, sys)
    
    def batch_predict(self, texts: List[str], task: str = 'language') -> List[Dict[str, Any]]:
        """
        Perform batch predictions on multiple texts
        
        Args:
            texts (List[str]): List of input texts
            task (str): Task type ('language', 'continent', or 'both')
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        try:
            if not texts:
                raise ValueError("Input texts list cannot be empty")
            
            logger.info(f"Starting batch prediction for {len(texts)} texts, task: {task}")
            
            results = []
            
            for i, text in enumerate(texts):
                try:
                    if task == 'language':
                        result = self.predict_language(text)
                    elif task == 'continent':
                        result = self.predict_continent(text)
                    elif task == 'both':
                        result = self.predict_both(text)
                    else:
                        raise ValueError("Task must be 'language', 'continent', or 'both'")
                    
                    result['batch_index'] = i
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to predict for text {i}: {str(e)}")
                    results.append({
                        'batch_index': i,
                        'input_text': text,
                        'error': str(e),
                        'prediction_failed': True
                    })
            
            logger.info(f"Batch prediction completed. Successful: {len([r for r in results if 'error' not in r])}/{len(texts)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise ModelPredictionError(e, sys)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models
        
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            if not self._models_loaded:
                return {'status': 'Models not loaded'}
            
            info = {
                'status': 'Models loaded successfully',
                'supported_languages': len(self.supported_languages),
                'language_list': self.supported_languages,
                'supported_continents': ['Africa', 'Asia', 'Europe', 'North America'],
                'models_available': {
                    'language_classification': True,
                    'continent_lda': True,
                    'continent_qda': True
                },
                'model_files_status': {
                    'language_pipeline': os.path.exists(self.config.language_pipeline_path),
                    'continent_lda': os.path.exists(self.config.continent_lda_model_path),
                    'continent_qda': os.path.exists(self.config.continent_qda_model_path),
                    'continent_vectorizer': os.path.exists(self.config.continent_vectorizer_path),
                    'continent_svd': os.path.exists(self.config.continent_svd_path),
                    'continent_label_encoder': os.path.exists(self.config.continent_label_encoder_path)
                }
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {'status': 'Error', 'error': str(e)}

class CustomData:
    """
    Custom data class for handling input data for predictions
    """
    
    def __init__(self, text: str):
        self.text = text
    
    def get_data_as_dict(self) -> Dict[str, str]:
        """
        Convert custom data to dictionary format
        
        Returns:
            Dict[str, str]: Data in dictionary format
        """
        try:
            return {'text': self.text}
        except Exception as e:
            raise CustomException(e, sys)

def create_prediction_pipeline() -> PredictionPipeline:
    """
    Factory function to create prediction pipeline
    
    Returns:
        PredictionPipeline: Initialized prediction pipeline
    """
    try:
        return PredictionPipeline()
    except Exception as e:
        logger.error(f"Failed to create prediction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Test prediction pipeline
    try:
        pipeline = PredictionPipeline()
        
        # Test predictions
        test_texts = [
            "Hello, how are you today?",
            "Bonjour, comment allez-vous?",
            "Hola, ¿cómo estás?",
            "Guten Tag, wie geht es Ihnen?",
            "Ciao, come stai?"
        ]
        
        print("Testing Prediction Pipeline")
        print("=" * 50)
        
        for text in test_texts:
            result = pipeline.predict_both(text)
            print(f"Text: {text}")
            print(f"Language: {result['language_prediction']['predicted_language']}")
            print(f"Continent: {result['continent_prediction']['predicted_continent']}")
            print(f"Consistent: {result['consistency_check']['consistent']}")
            print("-" * 30)
        
        print("\\nPrediction pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Prediction pipeline test failed: {str(e)}")
