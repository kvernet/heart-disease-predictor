from predictor import ProjectInfo, HeartDiseasePredictor


def create_example_patient():
    """Create an example patient for prediction"""
    return {
        'age': 52, 
        'sex': 1, 
        'cp': 3, 
        'trestbps': 125, 
        'chol': 212,
        'fbs': 0, 
        'restecg': 0, 
        'thalach': 168, 
        'exang': 0, 
        'oldpeak': 1.0, 
        'slope': 2, 
        'ca': 2, 
        'thal': 3
    }

# Then update the main function to include the project info:
def main():
    """Main function to run the heart disease prediction project"""
    
    # Display project and developer info
    ProjectInfo.display_header()
    ProjectInfo.display_developer_info()
    
    # Initialize the predictor
    predictor = HeartDiseasePredictor(models_dir='models', images_dir='images')
    
    try:
        # Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        data = predictor.load_and_preprocess_data()
        
        # Perform EDA
        print("\nStep 2: Performing Exploratory Data Analysis...")
        predictor.eda.generate_summary()
        predictor.eda.plot_target_distribution()
        predictor.eda.plot_distributions()
        predictor.eda.plot_correlation_heatmap()
        predictor.eda.plot_target_correlations()
        
        # Prepare features
        print("\nStep 3: Preparing features for modeling...")
        predictor.prepare_features(data)
        
        # Initialize and train models
        print("\nStep 4: Initializing machine learning models...")
        predictor.initialize_models()
        
        print("\nStep 5: Training models...")
        results = predictor.train_models()
        
        # Evaluate the best model
        print("\nStep 6: Evaluating the best model...")
        predictor.evaluate_best_model()
        
        # Save the model
        print("\nStep 7: Saving the model...")
        predictor.save_model('heart_disease_best_model.pkl')
        
        # Example prediction for a new patient
        print("\nStep 8: Making example prediction...")
        example_patient = create_example_patient()
        
        print(f"Example patient data: {example_patient}")
        prediction = predictor.predict_new_patient(example_patient)
        
        if prediction:
            print("\n" + "="*50)
            print("PREDICTION RESULTS")
            print("="*50)
            print(f"Diagnosis: {prediction['prediction']}")
            print(f"Probability: {prediction['probability']:.3f}")
            print(f"Risk Level: {prediction['risk_level']}")
            print(f"Confidence: {prediction['confidence']}")
            print("="*50)
        
        return predictor
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the complete project
    predictor = main()
    
    if predictor:
        print("\n🎉 Project completed successfully!")
        print(f"📁 Models saved in: {predictor.models_dir}")
        print(f"🖼️  Images saved in: {predictor.images_dir}")
        print("\n" + "="*70)
        print("Next steps:")
        print("1. Run 'streamlit run app.py' to start the web application")
        print("2. Open your browser to http://localhost:8501")
        print("3. Start making predictions!")
        print("="*70)
    else:
        print("\n❌ Project failed with errors!")