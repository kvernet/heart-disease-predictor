# ProjectInfo class
class ProjectInfo:
    """Project and Developer Information"""
    
    PROJECT_NAME = "Heart Disease Predictor System"
    VERSION = "1.0.0"
    DESCRIPTION = """
    A comprehensive machine learning system for predicting heart disease 
    using clinical data from the UCI Heart Disease dataset.
    """
    
    DEVELOPER_INFO = {
        'name': 'Kinson VERNET',
        'role': 'Machine Learning Specialist & Data Scientist',
        'github': 'https://github.com/kvernet',
        'linkedin': 'https://linkedin.com/in/kvernet',
        'email': 'kinson.vernet@gmail.com',
        'website': 'https://kvernet.com'
    }
    
    @classmethod
    def display_header(cls):
        """Display project header"""
        print("=" * 70)
        print(f"🤖 {cls.PROJECT_NAME}")
        print("=" * 70)
        print(f"Version: {cls.VERSION}")
        print(f"Description: {cls.DESCRIPTION.strip()}")
        print("=" * 70)
    
    @classmethod
    def display_developer_info(cls):
        """Display developer information"""
        print("\n" + "👨‍💻 DEVELOPED BY".center(70, '='))
        print(f"Name: {cls.DEVELOPER_INFO['name']}")
        print(f"Role: {cls.DEVELOPER_INFO['role']}")
        print(f"GitHub: {cls.DEVELOPER_INFO['github']}")
        print(f"LinkedIn: {cls.DEVELOPER_INFO['linkedin']}")
        print(f"Email: {cls.DEVELOPER_INFO['email']}")
        print(f"Website: {cls.DEVELOPER_INFO['website']}")
        print("=" * 70)

# Config class
class Config:
    """Configuration class for the project"""
    
    # Dataset URLs
    DATASET_URLS = {
        'cleveland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
        'hungarian': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
        'switzerland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
        'va': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'
    }
    
    # Column names based on the dataset documentation
    COLUMN_NAMES = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Feature descriptions
    FEATURE_DESCRIPTIONS = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)',
        'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
        'thal': 'Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)',
        'target': 'Diagnosis of heart disease (0: no disease, 1-4: presence of disease)'
    }
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Preprocessing
    NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']