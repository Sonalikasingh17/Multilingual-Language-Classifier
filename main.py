#!/usr/bin/myenv python3
"""
Main entry point for the Multilingual Language Classifier project.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline
from src.logger import get_logger

logger = get_logger(__name__)

def train_models():
    """Train all models using the training pipeline"""
    try:
        logger.info("Starting model training...")

        training_pipeline = TrainingPipeline()
        results = training_pipeline.start_training_pipeline()

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Language Classification Accuracy: {results['model_performance']['summary']['language_best_accuracy']:.2%}")
        print(f"Best Continent Classification: {results['model_performance']['summary']['continent_best_accuracy']:.2%}")
        print("="*60)

        return results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"Training failed: {str(e)}")
        sys.exit(1)

def predict_text(text, task='both'):
    """Make predictions on input text"""
    try:
        pipeline = PredictionPipeline()

        if task == 'language':
            result = pipeline.predict_language(text)
            print(f"\nLanguage: {result['predicted_language']}")
            print(f" Continent: {result['predicted_continent']}")

        elif task == 'continent':
            result = pipeline.predict_continent(text)
            print(f"\n Continent: {result['predicted_continent']}")

        else:  # both
            result = pipeline.predict_both(text)
            print(f"\n Language: {result['language_prediction']['predicted_language']}")
            print(f"   Continent: {result['continent_prediction']['predicted_continent']}")

        return result

    except Exception as e:
        logger.error(f" Prediction failed: {str(e)}")
        print(f" Prediction failed: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multilingual Language Classifier")

    parser.add_argument('command', choices=['train', 'predict', 'info', 'app'])
    parser.add_argument('input', nargs='?', help='Input text for prediction')
    parser.add_argument('--task', choices=['language', 'continent', 'both'], default='both')

    args = parser.parse_args()

    try:
        if args.command == 'train':
            train_models()

        elif args.command == 'predict':
            if not args.input:
                print(" Error: Please provide input text for prediction")
                sys.exit(1)
            predict_text(args.input, args.task)

        elif args.command == 'info':
            pipeline = PredictionPipeline()
            info = pipeline.get_model_info()
            print("\n🔍 Model Information:")
            print(f"Status: {info['status']}")

        elif args.command == 'app':
            print(" Launching Streamlit app...")
            os.system("streamlit run app.py")

    except Exception as e:
        print(f" Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
