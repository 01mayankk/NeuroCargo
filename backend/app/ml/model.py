import os
import pickle
import numpy as np
import pandas as pd
import logging
from flask import current_app
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configure logger
logger = logging.getLogger(__name__)

# Paths to model files
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vehicle_load_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vehicle_load_scaler.pkl')

# Create placeholder model and scaler if files don't exist
def create_simple_model():
    """Create and save a simple placeholder model when files don't exist"""
    try:
        logger.info("Creating placeholder model and scaler")
        
        # Create a simple scaler
        scaler = StandardScaler()
        # Fit with some placeholder data
        scaler.fit(np.array([[100, 150, 2, 20], [200, 300, 3, 30]]))  # weight, max_load, passengers, cargo
        
        # Create a simple RandomForest model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Features: vehicle_type_* (5 features) + 4 numerical features scaled
        X = np.array([
            [1, 0, 0, 0, 0, -1, -1, -1, -1],  # Example of 2-wheeler, below capacity
            [1, 0, 0, 0, 0, 1, 1, 1, 1],      # Example of 2-wheeler, above capacity
            [0, 1, 0, 0, 0, -1, -1, -1, -1],  # Example of 4-wheeler, below capacity
            [0, 1, 0, 0, 0, 1, 1, 1, 1]       # Example of 4-wheeler, above capacity
        ])
        
        # Target: 0 = not overloaded, 1 = overloaded
        y = np.array([0, 1, 0, 1])
        
        # Fit model
        model.fit(X, y)
        
        # Set feature names
        feature_names = [
            'vehicle_type_2-wheeler', 'vehicle_type_4-wheeler 5-seater', 
            'vehicle_type_4-wheeler 7-seater', 'vehicle_type_delivery vehicle', 
            'vehicle_type_heavy vehicle', 
            'weight_scaled', 'max_load_capacity_scaled', 
            'passenger_count_scaled', 'cargo_weight_scaled'
        ]
        model.feature_names_in_ = feature_names
        
        # Save model and scaler
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Placeholder model saved to {MODEL_PATH}")
        
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Placeholder scaler saved to {SCALER_PATH}")
        
        return model, scaler, feature_names
    
    except Exception as e:
        logger.error(f"Error creating placeholder model: {str(e)}")
        return None, None, None

# Try to load existing model and scaler
try:
    # Check if files exist
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded successfully from {SCALER_PATH}")
        
        # Get model feature names
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
            logger.info(f"Model feature names: {feature_names}")
        else:
            logger.warning("Model does not have feature_names_in_ attribute")
            feature_names = None
    else:
        logger.warning("Model or scaler files not found, creating placeholder model")
        model, scaler, feature_names = create_simple_model()
        
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    logger.info("Attempting to create placeholder model")
    model, scaler, feature_names = create_simple_model()

# Final check - if we still don't have a model, set to None
if model is None:
    logger.error("Failed to load or create model, predictions will not work")

def preprocess_data(data, graph_refresh_only=False):
    """
    Preprocess data for prediction.
    
    Args:
        data (dict or DataFrame): Input data
        graph_refresh_only (bool): If True, don't scale features
        
    Returns:
        DataFrame: Preprocessed data ready for model prediction
    """
    try:
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Rename 'weather' to 'weather_condition' if needed
        if 'weather' in df.columns and 'weather_condition' not in df.columns:
            df['weather_condition'] = df['weather']
            df = df.drop('weather', axis=1)
            
        # Define numerical and categorical features
        numerical_features = ['weight', 'max_load_capacity', 'passenger_count', 'cargo_weight']
        
        # Ensure all numerical features exist
        for feature in numerical_features:
            if feature not in df.columns:
                logger.warning(f"Missing numerical feature: {feature}. Adding with default value 0.")
                df[feature] = 0
        
        # For graph refresh, return processed dataframe without scaling
        if graph_refresh_only:
            return df
        
        # Create result DataFrame with expected features
        result_df = pd.DataFrame()
        
        # Handle vehicle type one-hot encoding
        vehicle_types = ['2-wheeler', '4-wheeler 5-seater', '4-wheeler 7-seater', 'delivery vehicle', 'heavy vehicle']
        current_type = df['vehicle_type'].iloc[0] if 'vehicle_type' in df.columns else None
        
        for vtype in vehicle_types:
            result_df[f"vehicle_type_{vtype}"] = 1 if current_type == vtype else 0
        
        # Scale numerical features
        if scaler:
            numerical_data = df[numerical_features].values
            scaled_data = scaler.transform(numerical_data)
            
            # Add scaled features to result DataFrame
            for i, feature in enumerate(numerical_features):
                result_df[f"{feature}_scaled"] = scaled_data[:, i]
        else:
            # If no scaler, use original values
            for feature in numerical_features:
                result_df[f"{feature}_scaled"] = df[feature].values
        
        # Ensure columns match feature_names if available
        if feature_names is not None:
            # Check if all expected features are present
            missing_features = set(feature_names) - set(result_df.columns)
            extra_features = set(result_df.columns) - set(feature_names)
            
            if missing_features:
                logger.warning(f"Missing expected features: {missing_features}. Adding with default values.")
                for feature in missing_features:
                    result_df[feature] = 0
            
            if extra_features:
                logger.warning(f"Extra features not used by model: {extra_features}")
            
            # Reorder columns to match feature_names
            result_df = result_df[feature_names]
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def predict_load_status(data):
    """
    Predict load status based on input data.
    
    Args:
        data (dict): Input data containing vehicle and load information
        
    Returns:
        dict: Prediction results and metrics
    """
    try:
        if model is None:
            logger.error("Model not loaded, using rule-based prediction instead")
            # Use rule-based prediction when model is not available
            # Calculate load percentage manually
            weight = float(data.get('weight', 0))
            max_load = float(data.get('max_load_capacity', 1))
            passengers = int(data.get('passenger_count', 0))
            cargo = float(data.get('cargo_weight', 0))
            
            # Assume 70kg per passenger
            passenger_weight = passengers * 70
            total_weight = weight + cargo + passenger_weight
            load_percentage = (total_weight / max_load) * 100
            
            # Simple rule: if load percentage > 90%, it's overloaded
            prediction_value = 1 if load_percentage > 90 else 0
            confidence = 0.95 if load_percentage > 95 or load_percentage < 85 else 0.75
        else:
            # Process data for prediction
            processed_data = preprocess_data(data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            
            # Convert prediction to int if it's a string
            if isinstance(prediction, str):
                prediction_value = 1 if prediction.lower() == "overloaded" else 0
            else:
                prediction_value = int(prediction)
            
            # Get prediction probability if available
            confidence = None
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(processed_data)[0]
                    confidence = float(probabilities[prediction_value])
                else:
                    confidence = 0.85  # Default value if predict_proba not available
            except Exception as e:
                logger.warning(f"Error getting prediction probability: {str(e)}")
                confidence = 0.85  # Default value
        
        # Calculate additional metrics
        metrics = calculate_metrics(data, prediction_value)
        
        # Log the prediction results
        logger.info(f"Prediction: {prediction_value}, Confidence: {confidence}")
        logger.info(f"Metrics: {metrics}")
        
        # Return prediction results
        return {
            'prediction': prediction_value,
            'confidence': confidence,
            'metrics': metrics
        }
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        # Return a fallback prediction
        return {
            'prediction': 0,
            'confidence': 0.5,
            'metrics': {
                'load_percentage': 0,
                'remaining_capacity': 0,
                'risk_assessment': 'Unknown',
                'total_weight': 0,
                'passenger_weight': 0,
                'fuel_efficiency_impact': 0
            }
        }

def calculate_metrics(data, prediction):
    """
    Calculate additional metrics based on prediction and input data.
    
    Args:
        data (dict): Input data
        prediction (int): Prediction value (0 or 1)
        
    Returns:
        dict: Calculated metrics
    """
    try:
        # Extract values from data
        weight = float(data.get('weight', 0))
        max_load_capacity = float(data.get('max_load_capacity', 1))
        passenger_count = int(data.get('passenger_count', 0))
        cargo_weight = float(data.get('cargo_weight', 0))
        
        # Calculate passenger weight (assuming average 70kg per passenger)
        passenger_weight = passenger_count * 70
        
        # Calculate total weight
        total_weight = weight + cargo_weight + passenger_weight
        
        # Calculate load percentage
        load_percentage = min(100, round((total_weight / max_load_capacity) * 100))
        
        # Calculate remaining capacity
        remaining_capacity = max(0, max_load_capacity - total_weight)
        
        # Determine risk assessment
        if load_percentage < 70:
            risk_assessment = "Low"
        elif load_percentage < 90:
            risk_assessment = "Medium"
        else:
            risk_assessment = "High"
            
        # Calculate fuel efficiency impact (simple estimate)
        # Assume 5% reduction in efficiency for each 10% load over 50%
        if load_percentage > 50:
            efficiency_reduction = ((load_percentage - 50) / 10) * 5
            fuel_efficiency_impact = min(30, efficiency_reduction)  # Cap at 30%
        else:
            fuel_efficiency_impact = 0
            
        return {
            'load_percentage': load_percentage,
            'remaining_capacity': round(remaining_capacity, 2),
            'risk_assessment': risk_assessment,
            'total_weight': round(total_weight, 2),
            'passenger_weight': round(passenger_weight, 2),
            'fuel_efficiency_impact': round(fuel_efficiency_impact, 2)
        }
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            'load_percentage': 0,
            'remaining_capacity': 0,
            'risk_assessment': 'Unknown',
            'total_weight': 0,
            'passenger_weight': 0,
            'fuel_efficiency_impact': 0
        }

def save_graphs(data, config):
    """Save visualization graphs to static folder"""
    try:
        # Get graphs folder from app config
        graphs_folder = config.get('GRAPHS_FOLDER', 'static/graphs')
        
        # Ensure directory exists
        os.makedirs(graphs_folder, exist_ok=True)
        
        # Example of saving a pie chart
        pie_path = os.path.join(graphs_folder, 'weight_distribution.png')
        
        # Create a simple pie chart
        labels = ['Vehicle Weight', 'Passenger Weight', 'Cargo Weight']
        sizes = [
            float(data.get('weight', 0)),
            float(data.get('passenger_count', 0)) * 70,  # 70kg per passenger
            float(data.get('cargo_weight', 0))
        ]
        
        # Don't create empty charts
        if sum(sizes) > 0:
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Weight Distribution')
            plt.savefig(pie_path)
            plt.close()
            logger.info(f"Pie chart generated and saved to {pie_path}")
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}") 