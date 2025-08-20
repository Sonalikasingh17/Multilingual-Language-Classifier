#!/usr/bin/env python3
"""
Multilingual Language Classifier Project Setup Script
This script creates the complete directory structure and all necessary files.
"""

import os

def create_directory_structure():
    """Create all required directories"""
    directories = [
        'src',
        'src/components', 
        'src/pipeline',
        'artifacts',
        'data',
        'data/raw',
        'data/processed', 
        'logs',
        'models',
        'notebooks',
        'reports',
        'static',
        'static/css',
        'static/images',
        'templates'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitkeep file to preserve empty directories
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('')

    print("✅ Directory structure created successfully!")

if __name__ == "__main__":
    print("🚀 Setting up Multilingual Language Classifier project...")
    create_directory_structure()
    print("🎉 Project setup completed!")
    print("\n📋 Next steps:")
    print("1. Copy all Python files to their respective directories")
    print("2. Run: python -m venv venv")
    print("3. Run: source venv/bin/activate (or venv\\Scripts\\activate on Windows)")
    print("4. Run: pip install -r requirements.txt")
    print("5. Run: pip install -e .")
    print("6. Run: python main.py train")
    print("7. Run: streamlit run app.py")
