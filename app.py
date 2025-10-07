import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffcccc;
        border-left: 5px solid #ff4b4b;
    }
    .medium-risk {
        background-color: #fff4cc;
        border-left: 5px solid #ffcc00;
    }
    .low-risk {
        background-color: #ccffcc;
        border-left: 5px solid #00cc00;
    }
    .feature-importance {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .developer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    .developer-links a {
        color: white;
        margin: 0 10px;
        text-decoration: none;
        font-weight: bold;
    }
    .developer-links a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

class HeartDiseaseApp:
    def __init__(self):
        self.models_dir = 'models'
        self.images_dir = 'images'
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_loaded = False
        
        # Developer Information
        self.developer_info = {
            'name': 'Kinson VERNET',
            'title': 'Machine Learning Specialist & Data Scientist',
            'github': 'https://github.com/kvernet',
            'linkedin': 'https://linkedin.com/in/kvernet',
            'website': 'https://kvernet.com',
            'email': 'kinson.vernet@gmail.com'
        }
        
        # Feature descriptions for tooltips
        self.feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = Male, 0 = Female)',
            'cp': 'Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)',
            'trestbps': 'Resting Blood Pressure (mm Hg)',
            'chol': 'Serum Cholesterol (mg/dl)',
            'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)',
            'restecg': 'Resting Electrocardiographic Results (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)',
            'thalach': 'Maximum Heart Rate Achieved',
            'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
            'oldpeak': 'ST Depression Induced by Exercise Relative to Rest',
            'slope': 'Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)',
            'ca': 'Number of Major Vessels Colored by Fluoroscopy (0-3)',
            'thal': 'Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)'
        }
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.path.join(self.models_dir, 'heart_disease_best_model.pkl')
            if os.path.exists(model_path):
                loaded_data = joblib.load(model_path)
                self.model = loaded_data['best_model']
                self.preprocessor = loaded_data['preprocessor']
                self.feature_names = loaded_data['feature_names']
                self.model_name = loaded_data['best_model_name']
                self.is_loaded = True
                return True
            else:
                st.error("Trained model not found. Please run the training script first.")
                return False
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict(self, patient_data):
        """Make prediction for patient data"""
        if not self.is_loaded:
            return None
        
        try:
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data], columns=self.feature_names)
            
            # Preprocess
            patient_processed = self.preprocessor.transform(patient_df)
            
            # Predict
            prediction = self.model.predict(patient_processed)[0]
            probability = self.model.predict_proba(patient_processed)[0][1]
            
            return {
                'prediction': prediction,
                'probability': probability,
                'diagnosis': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def create_input_form(self):
        """Create the input form for patient data"""
        st.sidebar.header("Patient Information")
        
        # Create two columns for better layout
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            age = st.slider("Age", 20, 100, 50, help=self.feature_descriptions['age'])
            trestbps = st.slider("Resting Blood Pressure", 90, 200, 120, help=self.feature_descriptions['trestbps'])
            chol = st.slider("Cholesterol", 100, 600, 200, help=self.feature_descriptions['chol'])
            thalach = st.slider("Max Heart Rate", 60, 220, 150, help=self.feature_descriptions['thalach'])
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1, help=self.feature_descriptions['oldpeak'])
        
        with col2:
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0], help=self.feature_descriptions['sex'])[1]
            cp = st.selectbox("Chest Pain Type", 
                            options=[("Typical Angina", 0), ("Atypical Angina", 1), 
                                   ("Non-anginal Pain", 2), ("Asymptomatic", 3)],
                            format_func=lambda x: x[0], help=self.feature_descriptions['cp'])[1]
            fbs = st.selectbox("Fasting Blood Sugar > 120", 
                             options=[("No", 0), ("Yes", 1)], 
                             format_func=lambda x: x[0], help=self.feature_descriptions['fbs'])[1]
            restecg = st.selectbox("Resting ECG", 
                                 options=[("Normal", 0), ("ST-T Wave Abnormality", 1), 
                                        ("Left Ventricular Hypertrophy", 2)],
                                 format_func=lambda x: x[0], help=self.feature_descriptions['restecg'])[1]
            exang = st.selectbox("Exercise Induced Angina", 
                               options=[("No", 0), ("Yes", 1)], 
                               format_func=lambda x: x[0], help=self.feature_descriptions['exang'])[1]
        
        # Additional features
        col3, col4 = st.sidebar.columns(2)
        with col3:
            slope = st.selectbox("ST Slope", 
                               options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                               format_func=lambda x: x[0], help=self.feature_descriptions['slope'])[1]
            ca = st.slider("Number of Major Vessels", 0, 3, 0, help=self.feature_descriptions['ca'])
        
        with col4:
            thal = st.selectbox("Thalassemia", 
                              options=[("Normal", 1), ("Fixed Defect", 2), ("Reversible Defect", 3)],
                              format_func=lambda x: x[0], help=self.feature_descriptions['thal'])[1]
        
        # Compile patient data
        patient_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        return patient_data
    
    def display_prediction_result(self, result):
        """Display prediction results"""
        if result is None:
            return
        
        # Determine risk class for styling
        risk_class = result['risk_level'].lower()
        
        st.markdown(f"""
        <div class="prediction-card {risk_class}-risk">
            <h2>Prediction Results</h2>
            <h3>Diagnosis: {result['diagnosis']}</h3>
            <p><strong>Probability:</strong> {result['probability']:.3f}</p>
            <p><strong>Risk Level:</strong> {result['risk_level']}</p>
            <p><strong>Model Used:</strong> {self.model_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display probability gauge
        self.display_probability_gauge(result['probability'])
    
    def display_probability_gauge(self, probability):
        """Display probability as a gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Probability"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_feature_importance(self):
        """Display feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            st.subheader("Feature Importance")
            
            importance = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            fig = px.bar(feature_importance_df.tail(10), 
                        x='importance', y='feature', 
                        orientation='h',
                        title='Top 10 Most Important Features',
                        labels={'importance': 'Importance', 'feature': 'Feature'})
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_developer_info(self):
        """Display developer information"""
        st.markdown("""
        <div class="developer-card">
            <h2>👨‍💻 Author</h2>
            <h3>{name}</h3>
            <p>{title}</p>
            <div class="developer-links">
                <a href="{github}" target="_blank">GitHub</a> |
                <a href="{linkedin}" target="_blank">LinkedIn</a> |
                <a href="{website}" target="_blank">Website</a> |
                <a href="mailto:{email}">Email</a>
            </div>
        </div>
        """.format(**self.developer_info), unsafe_allow_html=True)
    
    def display_model_info(self):
        """Display model information"""
        st.sidebar.header("Model Information")
        st.sidebar.info(f"**Model:** {self.model_name}")
        st.sidebar.info("This model predicts the likelihood of heart disease based on patient clinical data.")
        
        # Developer info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("About the Developer")
        st.sidebar.write(f"**{self.developer_info['name']}**")
        st.sidebar.write(self.developer_info['title'])
    
    def run(self):
        """Run the Streamlit application"""
        # Header
        st.markdown('<h1 class="main-header">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)
        
        if not self.is_loaded:
            st.error("""
            **Model not loaded!** Please ensure:
            1. You have run the training script first
            2. The model file exists in the 'models' directory
            3. The file is named 'heart_disease_best_model.pkl'
            """)
            return
        
        # Display model info in sidebar
        self.display_model_info()
        
        # Create input form in sidebar
        patient_data = self.create_input_form()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Heart Disease Prediction")
            
            if st.sidebar.button("Predict", type="primary"):
                with st.spinner("Analyzing patient data..."):
                    result = self.predict(patient_data)
                    self.display_prediction_result(result)
            
            # Display feature importance
            self.display_feature_importance()
        
        with col2:
            st.header("Quick Facts")
            st.info("""
            **About Heart Disease:**
            - Leading cause of death worldwide
            - Early detection saves lives
            - Regular check-ups are important
            - Lifestyle changes can reduce risk
            """)
            
            st.header("Risk Factors")
            st.warning("""
            High risk factors include:
            - High blood pressure
            - High cholesterol
            - Diabetes
            - Smoking
            - Obesity
            - Family history
            """)
        
        # Display developer info at the bottom
        self.display_developer_info()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Disclaimer:** This tool is for educational purposes only. "
            "Always consult with healthcare professionals for medical advice."
        )

def main():
    """Main function to run the Streamlit app"""
    app = HeartDiseaseApp()
    app.run()

if __name__ == "__main__":
    main()
