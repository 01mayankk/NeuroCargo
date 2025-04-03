import pickle
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("\n========== NEUROCARGO QUICK MODEL TRAINER ==========\n")

# Create output directories if they don't exist
os.makedirs('models', exist_ok=True)

# Define file paths
model_path = os.path.join('models', 'vehicle_load_model.pkl')
scaler_path = os.path.join('models', 'vehicle_load_scaler.pkl')
metrics_path = os.path.join('models', 'model_metrics.pkl')

print("Generating high-quality training dataset...")

# Define vehicle types
vehicle_types = ['2-wheeler', '4-wheeler 5-seater', '4-wheeler 7-seater', 'delivery vehicle', 'heavy vehicle']

# Create synthetic dataset with clear patterns (smaller size for speed)
n_samples = 1000
print(f"Generating {n_samples} training samples...")

# Create balanced samples
overloaded_data = []
not_overloaded_data = []

while len(overloaded_data) < n_samples//2 or len(not_overloaded_data) < n_samples//2:
    # Generate a vehicle with either overloaded or not overloaded status
    v_type = np.random.choice(vehicle_types)
    
    # Base properties
    if v_type == '2-wheeler':
        base_weight = np.random.uniform(100, 200)
        max_capacity = np.random.uniform(150, 250)
    elif v_type == '4-wheeler 5-seater':
        base_weight = np.random.uniform(800, 1500)
        max_capacity = np.random.uniform(400, 700)
    elif v_type == '4-wheeler 7-seater':
        base_weight = np.random.uniform(1500, 2500)
        max_capacity = np.random.uniform(600, 1000)
    elif v_type == 'delivery vehicle':
        base_weight = np.random.uniform(2500, 5000)
        max_capacity = np.random.uniform(2000, 5000)
    else:  # heavy vehicle
        base_weight = np.random.uniform(5000, 12000)
        max_capacity = np.random.uniform(8000, 20000)
    
    passengers = np.random.randint(1, 5)
    passenger_weight = passengers * 70
    
    # Create a sample that's definitely overloaded
    if len(overloaded_data) < n_samples//2:
        # Set cargo weight to guarantee overloading (120-150% of remaining capacity)
        remaining_capacity = max_capacity - passenger_weight
        cargo_weight = np.random.uniform(1.2, 1.5) * remaining_capacity
        
        total_weight = base_weight + passenger_weight + cargo_weight
        
        # Verify it's actually overloaded
        if total_weight > max_capacity:
            sample = {
                'vehicle_type': v_type,
                'weight': base_weight,
                'max_load_capacity': max_capacity,
                'passenger_count': passengers,
                'cargo_weight': cargo_weight,
                'overloaded': 'Overloaded'
            }
            overloaded_data.append(sample)
    
    # Create a sample that's definitely not overloaded
    if len(not_overloaded_data) < n_samples//2:
        # Set cargo weight to guarantee no overloading (50-80% of remaining capacity)
        remaining_capacity = max_capacity - passenger_weight
        cargo_weight = np.random.uniform(0.5, 0.8) * remaining_capacity
        
        total_weight = base_weight + passenger_weight + cargo_weight
        
        # Verify it's actually not overloaded
        if total_weight <= max_capacity:
            sample = {
                'vehicle_type': v_type,
                'weight': base_weight,
                'max_load_capacity': max_capacity,
                'passenger_count': passengers,
                'cargo_weight': cargo_weight,
                'overloaded': 'Not Overloaded'
            }
            not_overloaded_data.append(sample)

# Combine the datasets
data = overloaded_data[:n_samples//2] + not_overloaded_data[:n_samples//2]
np.random.shuffle(data)

# Convert to DataFrame
df = pd.DataFrame(data)
print("Dataset created with clear patterns.")
print(f"Class distribution: {df['overloaded'].value_counts().to_dict()}")

# Prepare features and target
X = df.drop('overloaded', axis=1)
y = df['overloaded']

# One-hot encode vehicle type
X_processed = pd.get_dummies(X, columns=['vehicle_type'])

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['weight', 'max_load_capacity', 'passenger_count', 'cargo_weight']
X_numerical = X_processed[numerical_cols].copy()
X_processed[numerical_cols] = scaler.fit_transform(X_numerical)

# Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining model...")
# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Overloaded')
recall = recall_score(y_test, y_pred, pos_label='Overloaded')
f1 = f1_score(y_test, y_pred, pos_label='Overloaded')
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Rename columns to match expected format
# Get feature names for numerical features
numerical_feature_names = [f"{col}_scaled" for col in numerical_cols]

# Rename columns
old_cols = X_processed.columns
new_cols = []
for col in old_cols:
    if col in numerical_cols:
        new_cols.append(f"{col}_scaled")
    else:
        new_cols.append(col)

X_processed.columns = new_cols

# Make sure the model has the feature names
if hasattr(model, 'feature_names_in_'):
    model.feature_names_in_ = np.array(new_cols)

# Save model, scaler, and metrics
print("\nSaving model and metrics...")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Saved model to {model_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved scaler to {scaler_path}")

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'confusion_matrix': cm,
    'feature_importance': dict(zip(new_cols, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
}

with open(metrics_path, 'wb') as f:
    pickle.dump(metrics, f)
print(f"Saved metrics to {metrics_path}")

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

print("\n========== FEATURE IMPORTANCE ==========")
if hasattr(model, 'feature_importances_'):
    importance_dict = dict(zip(new_cols, model.feature_importances_))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance:.4f}")

print("\n=================================================\n")
print("Model training complete!") 