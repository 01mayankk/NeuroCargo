import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample

print("\n========== NEUROCARGO MODEL TRAINER ==========\n")

# Create output directories if they don't exist
os.makedirs('models', exist_ok=True)

# Define file paths
model_path = os.path.join('models', 'vehicle_load_model.pkl')
scaler_path = os.path.join('models', 'vehicle_load_scaler.pkl')
metrics_path = os.path.join('models', 'model_metrics.pkl')

print("Generating high-quality training dataset...")

# Define vehicle types and environmental factors
vehicle_types = ['2-wheeler', '4-wheeler 5-seater', '4-wheeler 7-seater', 'delivery vehicle', 'heavy vehicle']
road_conditions = ['excellent', 'good', 'average', 'poor', 'very poor']
weather_conditions = ['clear', 'rainy', 'snowy', 'foggy', 'windy', 'extreme']
regions = ['urban', 'suburban', 'rural', 'highway', 'mountainous', 'coastal', 'desert']

# Environmental factor weightings
road_factors = {
    'excellent': 0,
    'good': 0.05,
    'average': 0.1,
    'poor': 0.2,
    'very poor': 0.3
}

weather_factors = {
    'clear': 0,
    'rainy': 0.15,
    'snowy': 0.25,
    'foggy': 0.2,
    'windy': 0.1,
    'extreme': 0.35
}

region_factors = {
    'urban': 0.05,
    'suburban': 0.03,
    'rural': 0.08,
    'highway': 0,
    'mountainous': 0.25,
    'coastal': 0.1,
    'desert': 0.15
}

# Create synthetic dataset with clear patterns and balanced classes
n_samples = 5000
print(f"Generating {n_samples} training samples...")

# First create balanced overloaded and non-overloaded cases
overloaded_samples = []
not_overloaded_samples = []

while len(overloaded_samples) < n_samples // 2 or len(not_overloaded_samples) < n_samples // 2:
    # Randomly select vehicle type
    v_type = np.random.choice(vehicle_types)
    
    # Generate properties based on vehicle type
    if v_type == '2-wheeler':
        weight = np.random.uniform(100, 200)
        max_load = np.random.uniform(150, 250)
        passengers = np.random.randint(1, 3)
    elif v_type == '4-wheeler 5-seater':
        weight = np.random.uniform(800, 1500)
        max_load = np.random.uniform(400, 700)
        passengers = np.random.randint(1, 6)
    elif v_type == '4-wheeler 7-seater':
        weight = np.random.uniform(1500, 2500)
        max_load = np.random.uniform(600, 1000)
        passengers = np.random.randint(1, 8)
    elif v_type == 'delivery vehicle':
        weight = np.random.uniform(2500, 5000)
        max_load = np.random.uniform(2000, 5000)
        passengers = np.random.randint(1, 4)
    else:  # heavy vehicle
        weight = np.random.uniform(5000, 12000)
        max_load = np.random.uniform(8000, 20000)
        passengers = np.random.randint(1, 3)
    
    # Generate environmental factors
    road_condition = np.random.choice(road_conditions)
    weather = np.random.choice(weather_conditions)
    region = np.random.choice(regions)
    
    # Calculate environmental impact
    road_factor = road_factors[road_condition]
    weather_factor = weather_factors[weather]
    region_factor = region_factors[region]
    env_factor = 1 + (road_factor + weather_factor + region_factor)
    
    # Calculate passenger weight
    passenger_weight = passengers * 70  # Assume 70kg per passenger
    
    # Calculate remaining capacity
    remaining_capacity = max_load - passenger_weight
    
    # Generate cargo weight
    if len(overloaded_samples) < n_samples // 2 and np.random.random() < 0.8:
        # Generate overloaded case (cargo weight that exceeds remaining capacity)
        # Set cargo weight to 110-150% of remaining capacity
        cargo_weight = np.random.uniform(1.1, 1.5) * remaining_capacity
        
        # Calculate total weight with environmental factors
        total_weight = weight + passenger_weight + cargo_weight
        adjusted_weight = total_weight * env_factor
        
        if adjusted_weight > max_load:  # Confirm it's actually overloaded
            sample = {
                'vehicle_type': v_type,
                'weight': weight,
                'max_load_capacity': max_load,
                'passenger_count': passengers,
                'cargo_weight': cargo_weight,
                'road_condition': road_condition,
                'weather': weather,
                'region': region,
                'overloaded': 'Overloaded'
            }
            overloaded_samples.append(sample)
    
    elif len(not_overloaded_samples) < n_samples // 2 and np.random.random() < 0.8:
        # Generate not overloaded case (cargo weight within remaining capacity)
        # Set cargo weight to 50-90% of remaining capacity
        cargo_weight = np.random.uniform(0.5, 0.9) * remaining_capacity
        
        # Calculate total weight with environmental factors
        total_weight = weight + passenger_weight + cargo_weight
        adjusted_weight = total_weight * env_factor
        
        if adjusted_weight <= max_load:  # Confirm it's actually not overloaded
            sample = {
                'vehicle_type': v_type,
                'weight': weight,
                'max_load_capacity': max_load,
                'passenger_count': passengers,
                'cargo_weight': cargo_weight,
                'road_condition': road_condition,
                'weather': weather,
                'region': region,
                'overloaded': 'Not Overloaded'
            }
            not_overloaded_samples.append(sample)

# Combine and shuffle samples
data = overloaded_samples[:n_samples//2] + not_overloaded_samples[:n_samples//2]
np.random.shuffle(data)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save the dataset for reference
df.to_csv('vehicle_load_data.csv', index=False)
print(f"Saved dataset to vehicle_load_data.csv")

# Check class distribution
class_distribution = df['overloaded'].value_counts()
print("Class distribution:")
print(class_distribution)

# Split data into train and test sets
X = df.drop('overloaded', axis=1)
y = df['overloaded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Define preprocessing for categorical and numerical features
categorical_features = ['vehicle_type', 'road_condition', 'weather', 'region']
numerical_features = ['weight', 'max_load_capacity', 'passenger_count', 'cargo_weight']

# Create preprocessing steps
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Create pipeline with preprocessing and model
print("\nTraining models and selecting the best one...")

# Define models to test
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Define hyperparameters to tune
    if name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
    else:  # GradientBoosting
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5, 10]
        }
    
    # Create grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Train model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Overloaded')
    recall = recall_score(y_test, y_pred, pos_label='Overloaded')
    f1 = f1_score(y_test, y_pred, pos_label='Overloaded')
    
    # Store results
    results[name] = {
        'model': best_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_params': grid_search.best_params_
    }
    
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")

# Select best model based on accuracy
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Extract final model from pipeline
final_model = best_model.named_steps['classifier']
final_preprocessor = best_model.named_steps['preprocessor']

# Get feature names
cat_features = final_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
num_features = numerical_features

# Combine feature names
feature_names = np.concatenate([cat_features, [f"{f}_scaled" for f in num_features]])

# Save preprocessed feature names with the model
if hasattr(final_model, 'feature_names_in_'):
    final_model.feature_names_in_ = feature_names

# Calculate confusion matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Overloaded')
recall = recall_score(y_test, y_pred, pos_label='Overloaded')
f1 = f1_score(y_test, y_pred, pos_label='Overloaded')

# Save model
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"Saved model to {model_path}")

# Save scaler (numerical transformer from the preprocessor)
with open(scaler_path, 'wb') as f:
    pickle.dump(final_preprocessor.named_transformers_['num'], f)
print(f"Saved scaler to {scaler_path}")

# Save metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': cm,
    'feature_importance': dict(zip(feature_names, final_model.feature_importances_)) if hasattr(final_model, 'feature_importances_') else {},
    'model_type': best_model_name,
    'best_params': results[best_model_name]['best_params'],
    'feature_names': list(feature_names)
}

with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)
print(f"Saved metrics to {metrics_path}")

# Display final metrics
print("\n========== FINAL MODEL PERFORMANCE METRICS ==========")
print(f"Model type: {best_model_name}")
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

# Feature importance
if hasattr(final_model, 'feature_importances_'):
    print("\n========== FEATURE IMPORTANCE ==========")
    feature_importances = final_model.feature_importances_
    
    # Create a DataFrame for sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Display top 10 features
    print("Top 10 most important features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

print("\n=================================================\n")

# Update the app.py file to use the new model
print("Updating app.py to use the new model...")

# Check if app.py exists
if os.path.exists('app.py'):
    with open('app.py', 'r') as f:
        app_code = f.read()
    
    # Update model path in app.py if needed
    if 'vehicle_load_model.pkl' in app_code and not os.path.join('models', 'vehicle_load_model.pkl') in app_code:
        app_code = app_code.replace('vehicle_load_model.pkl', os.path.join('models', 'vehicle_load_model.pkl'))
    
    # Update scaler path in app.py if needed
    if 'vehicle_load_scaler.pkl' in app_code and not os.path.join('models', 'vehicle_load_scaler.pkl') in app_code:
        app_code = app_code.replace('vehicle_load_scaler.pkl', os.path.join('models', 'vehicle_load_scaler.pkl'))
    
    # Write updated app.py
    with open('app.py', 'w') as f:
        f.write(app_code)
    
    print("Updated app.py with new model paths.")

print("Model training and setup complete!") 