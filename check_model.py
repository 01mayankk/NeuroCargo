import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("\n========== NEUROCARGO MODEL PERFORMANCE METRICS ==========\n")

# Define file paths
model_path = 'vehicle_load_model.pkl'
if not os.path.exists(model_path):
    model_path = os.path.join('models', 'vehicle_load_model.pkl')

scaler_path = 'vehicle_load_scaler.pkl'
if not os.path.exists(scaler_path):
    scaler_path = os.path.join('models', 'vehicle_load_scaler.pkl')

metrics_path = 'model_metrics.pkl'
if not os.path.exists(metrics_path):
    metrics_path = os.path.join('models', 'model_metrics.pkl')

# Load model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model).__name__}")
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load scaler
try:
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded successfully from: {scaler_path}")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# Check if metrics file exists
metrics_loaded = False
if os.path.exists(metrics_path):
    try:
        print(f"Found metrics file at: {metrics_path}")
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        
        # Print basic metrics
        print("\n========== ACTUAL MODEL PERFORMANCE METRICS ==========")
        print(f"Accuracy: {metrics.get('accuracy', 0) * 100:.2f}%")
        print(f"Precision: {metrics.get('precision', 0) * 100:.2f}%")
        print(f"Recall: {metrics.get('recall', 0) * 100:.2f}%")
        print(f"F1 Score: {metrics.get('f1', 0) * 100:.2f}%")
        
        # Cross-validation scores
        if 'cv_scores' in metrics:
            cv_scores = metrics['cv_scores']
            print(f"\nCross-validation scores: {', '.join([f'{score:.4f}' for score in cv_scores])}")
            print(f"Mean CV score: {np.mean(cv_scores):.4f}")
            print(f"Standard deviation: {np.std(cv_scores):.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print("\n========== CONFUSION MATRIX ==========")
            print("\nFormat: [[TN, FP], [FN, TP]]")
            print(f"\n{cm}")
            
            print("\nDetailed breakdown:")
            tn, fp, fn, tp = cm.ravel()
            print(f"True Negatives (correctly predicted as not overloaded): {tn}")
            print(f"False Positives (incorrectly predicted as overloaded): {fp}")
            print(f"False Negatives (incorrectly predicted as not overloaded): {fn}")
            print(f"True Positives (correctly predicted as overloaded): {tp}")
            
            # Calculate additional metrics from confusion matrix
            print("\n========== ADDITIONAL METRICS ==========")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"Specificity: {specificity:.4f}")
            
            positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"Positive Predictive Value: {positive_predictive_value:.4f}")
            
            negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
            print(f"Negative Predictive Value: {negative_predictive_value:.4f}")
        
        # Feature importance
        if 'feature_importance' in metrics:
            print("\n========== FEATURE IMPORTANCE ==========")
            feature_imp = metrics['feature_importance']
            if isinstance(feature_imp, dict):
                # Sort by importance value in descending order
                sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features:
                    print(f"{feature}: {importance:.4f}")
            else:
                print("Feature importance not in expected format")
                
        metrics_loaded = True
    except Exception as e:
        print(f"Error processing metrics from {metrics_path}: {e}")

# If metrics file doesn't exist, get expected features and evaluate model manually
if not metrics_loaded and model is not None:
    print("\nNo metrics file found or could not load it. Evaluating model with synthetic data...")
    
    # Get expected feature names from model
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        print(f"Model expects {len(expected_features)} features")
        print("Expected feature names:")
        for name in expected_features:
            print(f"- {name}")
    else:
        print("Model does not have feature_names_in_ attribute")
    
    # Create synthetic test data matching expected features
    try:
        # Define vehicle types
        vehicle_types = ['2-wheeler', '4-wheeler 5-seater', '4-wheeler 7-seater', 'delivery vehicle', 'heavy vehicle']
        
        # Define environmental factors
        road_conditions = ['excellent', 'good', 'average', 'poor', 'very poor']
        weather_conditions = ['clear', 'rainy', 'snowy', 'foggy', 'windy', 'extreme']
        regions = ['urban', 'suburban', 'rural', 'highway', 'mountainous', 'coastal', 'desert']
        
        # Create synthetic dataset - only needs properties for X processing
        n_samples = 500
        print(f"Generating {n_samples} test samples...")
        
        # Create a DataFrame with the right structure
        data = []
        for _ in range(n_samples):
            # Randomly select vehicle type
            v_type = np.random.choice(vehicle_types)
            
            # Generate base properties
            weight = np.random.uniform(100, 10000)
            max_load = np.random.uniform(200, 20000)
            passengers = np.random.randint(1, 7)
            cargo_weight = np.random.uniform(100, 15000)
            
            # Generate environmental factors
            road_condition = np.random.choice(road_conditions)
            weather = np.random.choice(weather_conditions)
            region = np.random.choice(regions)
            
            # Determine a realistic overloaded value
            total_weight = weight + (passengers * 70) + cargo_weight
            overloaded = 'Overloaded' if total_weight > max_load else 'Not Overloaded'
            
            # Create sample
            sample = {
                'vehicle_type': v_type,
                'weight': weight,
                'max_load_capacity': max_load,
                'passenger_count': passengers,
                'cargo_weight': cargo_weight,
                'road_condition': road_condition,
                'weather': weather,
                'region': region,
                'overloaded': overloaded
            }
            data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Train-test split for evaluation
        X = df.drop('overloaded', axis=1)  
        y = df['overloaded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create feature dataframe with exact columns in the right order
        if hasattr(model, 'feature_names_in_'):
            # Get basic features from the test data
            X_features = pd.DataFrame(index=X_test.index)
            
            # One-hot encode vehicle type - following model's expected features
            vehicle_type_features = [f for f in expected_features if f.startswith('vehicle_type_')]
            for feature in vehicle_type_features:
                vehicle = feature.replace('vehicle_type_', '')
                X_features[feature] = (X_test['vehicle_type'] == vehicle).astype(int)
            
            # Scale numerical features based on model's expected features
            numerical_features = ['weight', 'max_load_capacity', 'passenger_count', 'cargo_weight']
            scaled_features = [f for f in expected_features if any(f.startswith(nf) for nf in numerical_features)]
            
            if scaler:
                # Create a temporary DataFrame for scaling
                temp_df = X_test[numerical_features].copy()
                scaled_values = scaler.transform(temp_df)
                
                # Map scaled values to the right columns
                for i, feature in enumerate(numerical_features):
                    scaled_feature = f"{feature}_scaled"
                    if scaled_feature in expected_features:
                        X_features[scaled_feature] = scaled_values[:, i]
            
            # Verify all expected features are present
            missing_features = set(expected_features) - set(X_features.columns)
            if missing_features:
                print(f"Warning: Missing features in test data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X_features[feature] = 0
            
            # Ensure columns are in the right order
            X_features = X_features[expected_features]
            
            print(f"Created test features with shape: {X_features.shape}")
            
            # Make predictions
            print("Making predictions...")
            y_pred = model.predict(X_features)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label='Overloaded')
            recall = recall_score(y_test, y_pred, pos_label='Overloaded')
            f1 = f1_score(y_test, y_pred, pos_label='Overloaded')
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Save metrics to file
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importances = dict(zip(expected_features, model.feature_importances_))
                metrics['feature_importance'] = feature_importances
            
            # Save metrics
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"Saved model metrics to {metrics_path}")
            
            # Display metrics
            print("\n========== MODEL PERFORMANCE METRICS ==========")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Precision: {precision * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")
            
            # Display confusion matrix
            print("\n========== CONFUSION MATRIX ==========")
            print("\nFormat: [[TN, FP], [FN, TP]]")
            print(f"\n{cm}")
            
            print("\nDetailed breakdown:")
            tn, fp, fn, tp = cm.ravel()
            print(f"True Negatives (correctly predicted as not overloaded): {tn}")
            print(f"False Positives (incorrectly predicted as overloaded): {fp}")
            print(f"False Negatives (incorrectly predicted as not overloaded): {fn}")
            print(f"True Positives (correctly predicted as overloaded): {tp}")
            
            # Calculate additional metrics
            print("\n========== ADDITIONAL METRICS ==========")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"Specificity: {specificity:.4f}")
            
            positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"Positive Predictive Value: {positive_predictive_value:.4f}")
            
            negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
            print(f"Negative Predictive Value: {negative_predictive_value:.4f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                print("\n========== FEATURE IMPORTANCE ==========")
                importance_dict = dict(zip(expected_features, model.feature_importances_))
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features:
                    print(f"{feature}: {importance:.4f}")
        else:
            print("Cannot evaluate model without feature_names_in_ attribute")
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        import traceback
        traceback.print_exc()

# Check for model attributes
if model:
    print("\n========== MODEL ATTRIBUTES ==========")
    if hasattr(model, 'feature_names_in_'):
        print(f"\nNumber of features: {len(model.feature_names_in_)}")
        print("Feature names:")
        for i, name in enumerate(model.feature_names_in_):
            print(f"- {name}")
    elif hasattr(model, 'n_features_in_'):
        print(f"\nNumber of features: {model.n_features_in_}")
    
    if hasattr(model, 'n_estimators'):
        print(f"\nNumber of estimators: {model.n_estimators}")
    
    if hasattr(model, 'max_depth'):
        print(f"Max depth: {model.max_depth}")
    
    if hasattr(model, 'classes_'):
        print(f"\nClasses: {model.classes_}")

print("\n=================================================\n") 