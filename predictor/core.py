import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

from predictor import Config


# DataLoader class
class DataLoader:
    """Class to load and combine heart disease datasets"""
    
    def __init__(self):
        self.config = Config()
        self.data = None
        
    def load_dataset(self, url):
        """Load individual dataset from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Replace '?' with NaN and read data
            data_content = response.text.replace('?', 'NaN')
            df = pd.read_csv(StringIO(data_content), header=None, na_values='NaN')
            
            # Set column names
            if len(df.columns) <= len(self.config.COLUMN_NAMES):
                df.columns = self.config.COLUMN_NAMES[:len(df.columns)]
            
            return df
        except Exception as e:
            print(f"Error loading data from {url}: {e}")
            return None
    
    def load_all_datasets(self):
        """Load and combine all datasets"""
        datasets = []
        
        for location, url in self.config.DATASET_URLS.items():
            print(f"Loading {location} dataset...")
            df = self.load_dataset(url)
            if df is not None:
                df['location'] = location
                datasets.append(df)
        
        if datasets:
            self.data = pd.concat(datasets, ignore_index=True)
            print(f"Combined dataset shape: {self.data.shape}")
            return self.data
        else:
            raise Exception("Failed to load any datasets")
    
    def get_data(self):
        """Get the loaded data"""
        if self.data is None:
            self.load_all_datasets()
        return self.data

# DataPreprocessor class
class DataPreprocessor:
    """Class for data preprocessing and cleaning"""
    
    def __init__(self):
        self.config = Config()
        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False
        
    def preprocess_data(self, data):
        """Preprocess the heart disease data"""
        df = data.copy()
        
        # Convert target to binary (0: no disease, 1: disease)
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        
        # Impute numerical features with median
        numerical_imputer = SimpleImputer(strategy='median')
        df[self.config.NUMERICAL_FEATURES] = numerical_imputer.fit_transform(
            df[self.config.NUMERICAL_FEATURES]
        )
        
        # Impute categorical features with mode
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[self.config.CATEGORICAL_FEATURES] = categorical_imputer.fit_transform(
            df[self.config.CATEGORICAL_FEATURES]
        )
        
        return df
    
    def _feature_engineering(self, df):
        """Create new features"""
        
        # Age categories
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 40, 50, 60, 100], 
                                labels=['<40', '40-50', '50-60', '60+'])
        
        # Cholesterol categories
        df['chol_category'] = pd.cut(df['chol'],
                                   bins=[0, 200, 240, 1000],
                                   labels=['Normal', 'Borderline', 'High'])
        
        # Blood pressure categories
        df['bp_category'] = pd.cut(df['trestbps'],
                                 bins=[0, 120, 130, 140, 1000],
                                 labels=['Normal', 'Elevated', 'Stage1', 'Stage2'])
        
        # Heart rate zones
        df['hr_zone'] = pd.cut(df['thalach'],
                             bins=[0, 100, 120, 140, 160, 200, 300],
                             labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extreme'])
        
        return df
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.config.NUMERICAL_FEATURES),
                ('cat', categorical_transformer, self.config.CATEGORICAL_FEATURES)
            ]
        )
        
        return self.preprocessor
    
    def fit_preprocessor(self, X):
        """Fit the preprocessor on training data"""
        if self.preprocessor is None:
            self.create_preprocessor()
        
        self.preprocessor.fit(X)
        self.is_fitted = True
        return self.preprocessor
    
    def transform_features(self, X):
        """Transform features using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_preprocessor first.")
        
        return self.preprocessor.transform(X)

# EDA class
class EDA:
    """Class for Exploratory Data Analysis"""
    
    def __init__(self, data, images_dir='images'):
        self.data = data
        self.config = Config()
        self.images_dir = images_dir
        os.makedirs(self.images_dir, exist_ok=True)
    
    def generate_summary(self):
        """Generate comprehensive EDA summary"""
        
        print("=" * 50)
        print("HEART DISEASE DATASET - EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Target distribution
        print(f"\nTarget Distribution:")
        print(self.data['target_binary'].value_counts())
        print(f"\nDisease Prevalence: {self.data['target_binary'].mean():.2%}")
        
        # Statistical summary
        print(f"\nStatistical Summary:")
        print(self.data[self.config.NUMERICAL_FEATURES].describe())
    
    def plot_distributions(self, save_plots=True):
        """Plot feature distributions"""
        
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot numerical features
        numerical_features = self.config.NUMERICAL_FEATURES[:9]  # First 9 features
        
        for i, feature in enumerate(numerical_features):
            if i < len(axes):
                self.data[feature].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.images_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
            print(f"Saved feature distributions to {self.images_dir}/feature_distributions.png")
        
        plt.show()
    
    def plot_correlation_heatmap(self, save_plots=True):
        """Plot correlation heatmap"""
        
        plt.figure(figsize=(12, 8))
        
        # Select numerical features for correlation
        corr_features = self.config.NUMERICAL_FEATURES + ['target_binary']
        correlation_matrix = self.data[corr_features].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.images_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"Saved correlation heatmap to {self.images_dir}/correlation_heatmap.png")
        
        plt.show()
    
    def plot_target_correlations(self, save_plots=True):
        """Plot features vs target"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        features_to_plot = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'cp']
        
        for i, feature in enumerate(features_to_plot):
            if feature in self.data.columns and i < len(axes):
                self.data.boxplot(column=feature, by='target_binary', ax=axes[i])
                axes[i].set_title(f'{feature} vs Heart Disease')
        
        plt.suptitle('')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.images_dir}/target_correlations.png', dpi=300, bbox_inches='tight')
            print(f"Saved target correlations to {self.images_dir}/target_correlations.png")
        
        plt.show()
    
    def plot_target_distribution(self, save_plots=True):
        """Plot target variable distribution"""
        plt.figure(figsize=(8, 6))
        self.data['target_binary'].value_counts().plot(kind='bar')
        plt.title('Heart Disease Distribution')
        plt.xlabel('Heart Disease (0: No, 1: Yes)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        if save_plots:
            plt.savefig(f'{self.images_dir}/target_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved target distribution to {self.images_dir}/target_distribution.png")
        
        plt.show()

# HeartDiseasePredictor class
class HeartDiseasePredictor:
    """Main class for heart disease prediction"""
    
    def __init__(self, models_dir='models', images_dir='images'):
        self.config = Config()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.eda = None
        self.models = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importance = None
        self.is_trained = False
        
        # Create directories
        self.models_dir = models_dir
        self.images_dir = images_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess data"""
        print("Loading data...")
        data = self.data_loader.get_data()
        
        print("Preprocessing data...")
        processed_data = self.preprocessor.preprocess_data(data)
        
        self.eda = EDA(processed_data, self.images_dir)
        
        return processed_data
    
    def prepare_features(self, data):
        """Prepare features and target for modeling"""
        
        # Define features and target
        features = self.config.NUMERICAL_FEATURES + self.config.CATEGORICAL_FEATURES
        X = data[features]
        y = data['target_binary']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, stratify=y
        )
        
        # Create and fit preprocessor
        self.preprocessor.create_preprocessor()
        self.preprocessor.fit_preprocessor(self.X_train)
        self.X_train_processed = self.preprocessor.transform_features(self.X_train)
        self.X_test_processed = self.preprocessor.transform_features(self.X_test)
        
        # Store feature names
        self.feature_names = (self.config.NUMERICAL_FEATURES + 
                             self.config.CATEGORICAL_FEATURES)
        
        print(f"Training set shape: {self.X_train_processed.shape}")
        print(f"Test set shape: {self.X_test_processed.shape}")
        
        return self.X_train_processed, self.X_test_processed, self.y_train, self.y_test
    
    def initialize_models(self):
        """Initialize machine learning models"""
        
        # Compute class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                class_weight=class_weight_dict
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.config.RANDOM_STATE,
                class_weight=class_weight_dict
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.config.RANDOM_STATE,
                class_weight=class_weight_dict,
                max_iter=1000
            )
        }
        
        return self.models
    
    def train_models(self):
        """Train all models and evaluate performance"""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_processed, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test_processed)
            y_pred_proba = model.predict_proba(self.X_test_processed)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_processed, self.y_train, 
                                      cv=self.config.CV_FOLDS, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.results = results
        self._select_best_model()
        self.is_trained = True
        
        return results
    
    def _select_best_model(self):
        """Select the best model based on AUC score"""
        best_score = -1
        best_model_name = None
        
        for name, result in self.results.items():
            if result['auc_score'] > best_score:
                best_score = result['auc_score']
                best_model_name = name
        
        self.best_model = self.results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"AUC Score: {best_score:.4f}")
        print(f"{'='*50}")
    
    def evaluate_best_model(self, save_plots=True):
        """Comprehensive evaluation of the best model"""
        
        if not self.is_trained:
            print("No model trained yet. Please train models first.")
            return
        
        best_result = self.results[self.best_model_name]
        
        print(f"\nDetailed Evaluation for {self.best_model_name}:")
        print("=" * 60)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_result['predictions']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, best_result['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_plots:
            plt.savefig(f'{self.images_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {self.images_dir}/confusion_matrix.png")
        
        plt.show()
        
        # ROC Curve
        self.plot_roc_curve(save_plots=save_plots)
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance(save_plots=save_plots)
    
    def plot_roc_curve(self, save_plots=True):
        """Plot ROC curve for all models"""
        
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            auc_score = result['auc_score']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_plots:
            plt.savefig(f'{self.images_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
            print(f"Saved ROC curves to {self.images_dir}/roc_curves.png")
        
        plt.show()
    
    def plot_feature_importance(self, top_n=10, save_plots=True):
        """Plot feature importance for tree-based models"""
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.bar(range(min(top_n, len(importance))), 
                   importance[indices][:top_n])
            plt.xticks(range(min(top_n, len(importance))), 
                      [self.feature_names[i] for i in indices[:top_n]], 
                      rotation=45)
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f'{self.images_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
                print(f"Saved feature importance to {self.images_dir}/feature_importance.png")
            
            plt.show()
            
            self.feature_importance = dict(zip(
                [self.feature_names[i] for i in indices],
                importance[indices]
            ))
    
    def predict_new_patient(self, patient_data):
        """Predict heart disease for a new patient"""
        
        if not self.is_trained:
            print("No trained model available. Please train the model first.")
            return None
        
        if not self.preprocessor.is_fitted:
            print("Preprocessor not fitted. Please train the model first.")
            return None
        
        # Convert patient data to DataFrame
        patient_df = pd.DataFrame([patient_data], columns=self.feature_names)
        
        # Preprocess the data using the already fitted preprocessor
        try:
            patient_processed = self.preprocessor.transform_features(patient_df)
            
            # Make prediction
            prediction = self.best_model.predict(patient_processed)[0]
            probability = self.best_model.predict_proba(patient_processed)[0][1]
            
            result = {
                'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
                'probability': probability,
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
                'confidence': 'High' if probability > 0.8 or probability < 0.2 else 'Medium'
            }
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def save_model(self, filename='heart_disease_model.pkl'):
        """Save the trained model"""
        if self.best_model is not None and self.preprocessor.is_fitted:
            model_data = {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'preprocessor': self.preprocessor.preprocessor,
                'feature_names': self.feature_names,
                'config': self.config,
                'is_trained': self.is_trained,
                'preprocessor_fitted': self.preprocessor.is_fitted
            }
            filepath = os.path.join(self.models_dir, filename)
            joblib.dump(model_data, filepath)
            print(f"Model saved as {filepath}")
        else:
            print("No trained model to save.")
    
    def load_model(self, filename='heart_disease_model.pkl'):
        """Load a trained model"""
        try:
            filepath = os.path.join(self.models_dir, filename)
            loaded_data = joblib.load(filepath)
            self.best_model = loaded_data['best_model']
            self.best_model_name = loaded_data['best_model_name']
            self.preprocessor.preprocessor = loaded_data['preprocessor']
            self.preprocessor.is_fitted = loaded_data['preprocessor_fitted']
            self.feature_names = loaded_data['feature_names']
            self.is_trained = loaded_data['is_trained']
            print(f"Model loaded from {filepath}")
            print(f"Loaded model: {self.best_model_name}")
        except FileNotFoundError:
            print(f"Model file {filename} not found.")
        except Exception as e:
            print(f"Error loading model: {e}")