import os
import sys
import pandas as pd
from typing import Dict, Any

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

class TrainingPipeline:
    """
    Complete end-to-end training pipeline for multilingual language classification
    """
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def start_training_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline
        
        Returns:
            Dict[str, Any]: Training results and performance metrics
        """
        try:
            logger.info("🚀 Starting Multilingual Language Classification Training Pipeline")
            logger.info("=" * 80)
            
            # Step 1: Data Ingestion
            logger.info("📥 STEP 1: DATA INGESTION")
            train_data_path, val_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            # Load the ingested data
            train_df = pd.read_csv(train_data_path)
            val_df = pd.read_csv(val_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logger.info(f"✅ Data ingestion completed successfully")
            logger.info(f"   📊 Train samples: {len(train_df):,}")
            logger.info(f"   📊 Validation samples: {len(val_df):,}")
            logger.info(f"   📊 Test samples: {len(test_df):,}")
            logger.info(f"   🌍 Languages: {train_df['locale'].nunique()}")
            logger.info(f"   🗺️  Continents: {train_df['continent'].nunique()}")
            
            # Step 2: Data Transformation
            logger.info("\\n🔄 STEP 2: DATA TRANSFORMATION")
            transformed_data = self.data_transformation.transform_data(train_df, val_df, test_df)
            
            logger.info("✅ Data transformation completed successfully")
            logger.info(f"   🎯 Language features shape: {transformed_data['language']['train'][0].shape}")
            logger.info(f"   🎯 Continent features shape: {transformed_data['continent']['train'][0].shape}")
            
            # Step 3: Model Training
            logger.info("\\n🤖 STEP 3: MODEL TRAINING")
            performance_metrics = self.model_trainer.initiate_model_trainer(transformed_data)
            
            # Step 4: Train complete language pipeline for inference
            logger.info("\\n🔗 STEP 4: CREATING INFERENCE PIPELINE")
            pipeline_performance = self.model_trainer.train_complete_language_pipeline(train_df, val_df, test_df)
            
            # Combine all results
            training_results = {
                'status': 'SUCCESS',
                'data_info': {
                    'train_samples': len(train_df),
                    'validation_samples': len(val_df),
                    'test_samples': len(test_df),
                    'total_samples': len(train_df) + len(val_df) + len(test_df),
                    'languages': train_df['locale'].nunique(),
                    'continents': train_df['continent'].nunique(),
                    'language_list': sorted(train_df['locale'].unique().tolist()),
                    'continent_list': sorted(train_df['continent'].unique().tolist())
                },
                'model_performance': performance_metrics,
                'pipeline_performance': pipeline_performance,
                'artifacts_created': {
                    'data_files': [train_data_path, val_data_path, test_data_path],
                    'model_files': [
                        self.model_trainer.model_trainer_config.language_model_file_path,
                        self.model_trainer.model_trainer_config.continent_lda_model_file_path,
                        self.model_trainer.model_trainer_config.continent_qda_model_file_path,
                        self.model_trainer.model_trainer_config.language_pipeline_path
                    ],
                    'preprocessor_files': [
                        self.data_transformation.data_transformation_config.language_vectorizer_path,
                        self.data_transformation.data_transformation_config.continent_vectorizer_path,
                        self.data_transformation.data_transformation_config.continent_svd_path,
                        self.data_transformation.data_transformation_config.label_encoder_path,
                        self.data_transformation.data_transformation_config.continent_label_encoder_path
                    ]
                }
            }
            
            # Final summary
            logger.info("\\n" + "=" * 80)
            logger.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info("📈 FINAL PERFORMANCE SUMMARY:")
            logger.info(f"   🏆 Language Classification Accuracy: {performance_metrics['summary']['language_best_accuracy']:.2%}")
            logger.info(f"   🏆 Continent LDA Accuracy: {performance_metrics['summary']['continent_lda_accuracy']:.2%}")
            logger.info(f"   🏆 Continent QDA Accuracy: {performance_metrics['summary']['continent_qda_accuracy']:.2%}")
            logger.info(f"   🏆 Pipeline Accuracy: {pipeline_performance['test_accuracy']:.2%}")
            logger.info("=" * 80)
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

def run_training_pipeline():
    """
    Convenience function to run the training pipeline
    """
    try:
        pipeline = TrainingPipeline()
        results = pipeline.start_training_pipeline()
        return results
    except Exception as e:
        logger.error(f"Failed to run training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info("🌟 Starting Multilingual Language Classification Training")
        results = run_training_pipeline()
        logger.info("🌟 Training completed successfully!")
        
        # Print key results
        print("\\n" + "="*60)
        print("TRAINING RESULTS SUMMARY")
        print("="*60)
        print(f"Total samples processed: {results['data_info']['total_samples']:,}")
        print(f"Languages supported: {results['data_info']['languages']}")
        print(f"Continents covered: {results['data_info']['continents']}")
        print(f"Language classification accuracy: {results['model_performance']['summary']['language_best_accuracy']:.2%}")
        print(f"Best continent classification accuracy: {results['model_performance']['summary']['continent_best_accuracy']:.2%}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)
