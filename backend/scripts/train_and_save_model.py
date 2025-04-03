import os
import sys
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set paths for model and data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vehicle_load_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'vehicle_load_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'vehicle_load_scaler.pkl')

def load_data(file_path):
    """Load data from CSV file"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess the data for model training"""
    try:
        logger.info("Preprocessing data...")
        
        # Check if required columns exist
        required_columns = ['vehicle_type', 'weight', 'max_load_capacity', 
                           'passenger_count', 'cargo_weight', 'is_overloaded']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column missing in data: {col}")
                raise ValueError(f"Required column missing in data: {col}")
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype == 'object' or col == 'vehicle_type':
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # One-hot encode vehicle type
        if 'vehicle_type' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['vehicle_type'], prefix='vehicle_type')
        
        # Define features and target
        X = df_processed.drop('is_overloaded', axis=1)
        y = df_processed['is_overloaded']
        
        # Convert target to integers if needed
        if y.dtype == 'object':
            y = y.map({'True': 1, 'False': 0, True: 1, False: 0, 'Overloaded': 1, 'Safe': 0})
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a scaler
        scaler = StandardScaler()
        
        # Get numerical features (exclude one-hot encoded columns)
        numerical_features = [col for col in X_train.columns if not col.startswith('vehicle_type_')]
        
        # Scale numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
        
        logger.info("Data preprocessing completed successfully")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def train_model(X_train, y_train):
    """Train the model"""
    try:
        logger.info("Training model...")
        
        # Create a RandomForest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save feature names
        model.feature_names_in_ = X_train.columns.tolist()
        
        logger.info("Model training completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics"""
    try:
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logger.info("Top 10 important features:")
        logger.info(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def save_model(model, scaler):
    """Save the model and scaler to disk"""
    try:
        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {SCALER_PATH}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    """Main function to run the training pipeline"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
        
        # Check if data file exists, and create sample data if it doesn't
        if not os.path.exists(DATA_PATH):
            logger.warning(f"Data file not found: {DATA_PATH}")
            logger.info("Creating sample training data...")
            create_sample_data(DATA_PATH)
            
        # Load data
        df = load_data(DATA_PATH)
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model and scaler
        save_model(model, scaler)
        
        logger.info("Model training and saving completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        sys.exit(1)

def create_sample_data(file_path):
    """Create a sample dataset for demonstration purposes"""
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define vehicle types
        vehicle_types = ['2-wheeler', '4-wheeler 5-seater', '4-wheeler 7-seater', 'delivery vehicle', 'heavy vehicle']
        
        # Define sample size
        n_samples = 5000
        
        # Create dataframe
        data = {
            'vehicle_type': np.random.choice(vehicle_types, size=n_samples),
            'weight': np.random.uniform(100, 5000, n_samples),
            'max_load_capacity': np.random.uniform(200, 10000, n_samples),
            'passenger_count': np.random.randint(0, 10, n_samples),
            'cargo_weight': np.random.uniform(0, 5000, n_samples),
            'weather_condition': np.random.choice(['normal', 'rainy', 'snowy', 'windy', 'stormy'], size=n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate a realistic overloaded condition
        passenger_weight = df['passenger_count'] * 70  # Assume average 70kg per passenger
        total_weight = df['weight'] + df['cargo_weight'] + passenger_weight
        load_percentage = (total_weight / df['max_load_capacity']) * 100
        
        # Define overloaded based on percentage and add some randomness
        df['is_overloaded'] = (load_percentage > 90) | ((load_percentage > 80) & (np.random.random(n_samples) > 0.7))
        
        # Add some correlations with weather
        weather_factor = df['weather_condition'].map({
            'normal': 1.0,
            'rainy': 1.1,
            'snowy': 1.2,
            'windy': 1.15,
            'stormy': 1.25
        })
        adjusted_load = load_percentage * weather_factor
        df.loc[adjusted_load > 95, 'is_overloaded'] = True
        
        # Save to CSV
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Sample data created and saved to {file_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating sample data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 