import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from src.pipeline.predict_pipeline import PredictionPipeline, create_prediction_pipeline
from src.pipeline.train_pipeline import TrainingPipeline

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Configure page
st.set_page_config(
    page_title="Multilingual Language Classifier",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>      
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: olive !important;
        margin-bottom: 2rem;
        user-select: none;
}
    .feature-card { 
        background:  #e5efd2;          /* Light neutral background */
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        color: #222;                 /* Dark, readable text */
        font-size: 1.2rem;
    }
    .feature-card h3 {
        margin-top: 0;
        color: #332888;              /* Optional: feature title color */
    }
           
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.title("Navigation 📍")
    page = st.sidebar.selectbox("Choose a page 👇", ["🏠 Home", "🔮 Prediction", "🛠️ Training"])

    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page()
    elif page == "🛠️ Training":
        show_training_page()

def show_home_page():
    st.markdown(
        '<h1 class="main-header"  style="color: olive;">🌐 Multilingual Language Classifier 🌐</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### 🧠 Advanced AI-powered language and continent classification using the MASSIVE dataset

    This application can classify text into:
    - 🌍 **27 Roman-script languages** with 98%+ accuracy
    - 🗺️ **4 continents** (Europe, Asia, Africa, North America)
    """)


    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 High Accuracy</h3>
            <p> 98%+ accuracy on language classification </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""                   
        <div class="feature-card">
            <h3>🌍 27 Languages</h3>
            <p>Supports Roman-script languages  </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Fast Processing</h3>
            <p>  Real-time predictions  </p>
        </div>
        """, unsafe_allow_html=True)



def show_prediction_page():
    st.markdown('<h1 class="main-header" style="color: olive;">🔮 Text Prediction</h1>', unsafe_allow_html=True)

    # Initialize prediction pipeline
    if 'prediction_pipeline' not in st.session_state:
        try:
            with st.spinner("Loading models..."):
                st.session_state.prediction_pipeline = create_prediction_pipeline()
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please train the models first using the Training page.")
            return

    # Input section
    user_text = st.text_area("Enter text to classify:", height=150)

    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox("Analysis Type:", ["Both Language & Continent", "Language Only", "Continent Only"])
    with col2:
        if "Continent" in task_type:
            continent_model = st.selectbox("Continent Model:", ["LDA", "QDA"])

    if st.button("🚀 Analyze Text", type="primary"):
        if user_text.strip():
            try:
                with st.spinner("Analyzing..."):
                    pipeline = st.session_state.prediction_pipeline

                    if task_type == "Language Only":
                        result = pipeline.predict_language(user_text)
                        st.success(f"**Language:** {result['predicted_language']}")
                        st.info(f"**Continent:** {result['predicted_continent']}")

                    elif task_type == "Continent Only":
                        result = pipeline.predict_continent(user_text, continent_model.lower())
                        st.success(f"**Continent:** {result['predicted_continent']}")
                        st.info(f"**Model:** {result['model_used']}")

                    else:  # Both
                        result = pipeline.predict_both(user_text, continent_model.lower())

                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"**Language:** {result['language_prediction']['predicted_language']}")
                        with col2:
                            st.success(f"**Continent:** {result['continent_prediction']['predicted_continent']}")

                        # Consistency check
                        consistent = result['consistency_check']['consistent']
                        if consistent:
                            st.success("✅ Predictions are consistent!")
                        else:
                            st.warning("⚠️ Predictions are inconsistent")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

def show_training_page():
    st.markdown('<h1 class="main-header" style="color: olive;">🛠️ Model Training</h1>', unsafe_allow_html=True)

    st.warning("⚠️ Training requires significant computational resources and time.")

    st.markdown("""
    ## Training Pipeline Overview

    1. **Data Ingestion**: Load MASSIVE dataset
    2. **Data Transformation**: Create features
    3. **Model Training**: Train ML models
    4. **Evaluation**: Test performance
    """)

    if st.button("🚀 Start Training", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Training in progress..."):
                training_pipeline = TrainingPipeline()
                results = training_pipeline.start_training_pipeline()

                progress_bar.progress(100)
                status_text.text("✅ Training completed!")

                st.success("🎉 Training completed successfully!")

                # Show results
                perf = results['model_performance']['summary']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Language Accuracy", f"{perf['language_best_accuracy']:.2%}")
                with col2:
                    st.metric("Continent LDA", f"{perf['continent_lda_accuracy']:.2%}")
                with col3:
                    st.metric("Continent QDA", f"{perf['continent_qda_accuracy']:.2%}")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
    main()
