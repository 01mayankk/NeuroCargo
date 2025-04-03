# Import necessary libraries for data manipulation, numerical operations, randomization, and file operations
import pandas as pd  # Data manipulation and analysis library
import numpy as np  # Numerical computing library
import random  # Random number and selection generation
import os  # Operating system interactions for file and path operations
import datetime

# Import specialized libraries for machine learning preprocessing
from imblearn.over_sampling import SMOTE, ADASYN  # Synthetic Minority Over-sampling Technique for balancing datasets
from sklearn.preprocessing import StandardScaler, RobustScaler  # Feature scaling utility
import yaml

# Define the filename for the generated dataset
dataset_filename = "vehicle_data.csv"
metadata_filename = "dataset_metadata.yaml"

# Check if the dataset file already exists to avoid regenerating
if not os.path.exists(dataset_filename):
    # Set the number of entries to generate in the dataset
    num_entries = 20000  # Increased from 10,000 to 20,000 for better training
    data = []
    
    # Generation timestamp for versioning
    generation_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Weather conditions and their probability weights
    weather_conditions = ["Clear", "Rain", "Snow", "Fog"]
    weather_weights = [0.7, 0.15, 0.05, 0.1]
    
    # Road conditions and their probability weights
    road_conditions = ["Good", "Fair", "Poor"]
    road_weights = [0.6, 0.3, 0.1]
    
    # Regions with different regulations
    regions = ["Urban", "Rural", "Highway", "Mountain"]

    # Generate synthetic vehicle data entries
    for _ in range(num_entries):
        # Generate seasonal and time-based features
        month = random.randint(1, 12)
        is_winter = 1 if month in [12, 1, 2] else 0
        is_summer = 1 if month in [6, 7, 8] else 0
        
        # Time of day affects loading patterns
        hour_of_day = random.randint(0, 23)
        is_peak_hours = 1 if hour_of_day in [7, 8, 9, 16, 17, 18] else 0
        
        # Region affects loading regulations
        region = random.choice(regions)
        
        # Environmental factors
        temperature = random.uniform(-10, 40)  # Temperature in Celsius
        weather = random.choices(weather_conditions, weights=weather_weights)[0]
        road_condition = random.choices(road_conditions, weights=road_weights)[0]
        
        # Randomly select a vehicle type with more realistic distribution
        vehicle_type_options = ["2-wheeler", "4-wheeler 5-seater", "4-wheeler 7-seater", "delivery vehicle", "heavy vehicle"]
        vehicle_type_weights = [0.2, 0.4, 0.2, 0.15, 0.05]  # More 4-wheelers than heavy vehicles
        vehicle_type = random.choices(vehicle_type_options, weights=vehicle_type_weights)[0]

        # Generate data specific to each vehicle type with realistic constraints
        if vehicle_type == "2-wheeler":
            # Vehicle dimensions in meters (length, width, height)
            length = round(random.uniform(1.8, 2.2), 2)
            width = round(random.uniform(0.7, 0.9), 2)
            height = round(random.uniform(1.0, 1.3), 2)
            
            # Volume calculation in cubic meters
            volume = round(length * width * height, 2)
            
            max_capacity = random.randint(200, 350)  # Maximum load capacity range
            empty_weight = random.randint(80, 120)  # Vehicle's base weight
            passenger_count = random.randint(1, 2)  # Number of passengers
            passenger_weight = passenger_count * random.randint(60, 90)  # Variable passenger weight
            cargo_weight = random.randint(0, 50)  # Cargo weight range
            
            # Maintenance factors
            tire_pressure = round(random.uniform(28, 36), 1)  # PSI
            maintenance_score = random.randint(1, 10)  # 1-10 scale
            
            # Load distribution (front/rear) in percentage
            front_weight_percent = round(random.uniform(40, 60), 1)
            rear_weight_percent = round(100 - front_weight_percent, 1)
            
        elif vehicle_type == "4-wheeler 5-seater":
            length = round(random.uniform(4.2, 4.8), 2)
            width = round(random.uniform(1.7, 1.9), 2)
            height = round(random.uniform(1.4, 1.6), 2)
            volume = round(length * width * height, 2)
            
            max_capacity = random.randint(800, 1200)
            empty_weight = random.randint(600, 800)
            passenger_count = random.randint(1, 5)
            passenger_weight = passenger_count * random.randint(60, 90)
            cargo_weight = random.randint(0, 150)
            
            tire_pressure = round(random.uniform(30, 38), 1)
            maintenance_score = random.randint(1, 10)
            
            front_weight_percent = round(random.uniform(45, 55), 1)
            rear_weight_percent = round(100 - front_weight_percent, 1)
            
        elif vehicle_type == "4-wheeler 7-seater":
            length = round(random.uniform(4.5, 5.2), 2)
            width = round(random.uniform(1.8, 2.0), 2)
            height = round(random.uniform(1.6, 1.8), 2)
            volume = round(length * width * height, 2)
            
            max_capacity = random.randint(1000, 1500)
            empty_weight = random.randint(800, 1000)
            passenger_count = random.randint(1, 7)
            passenger_weight = passenger_count * random.randint(60, 90)
            cargo_weight = random.randint(0, 200)
            
            tire_pressure = round(random.uniform(30, 38), 1)
            maintenance_score = random.randint(1, 10)
            
            front_weight_percent = round(random.uniform(40, 50), 1)
            rear_weight_percent = round(100 - front_weight_percent, 1)
            
        elif vehicle_type == "delivery vehicle":
            length = round(random.uniform(5.0, 7.0), 2)
            width = round(random.uniform(2.0, 2.5), 2)
            height = round(random.uniform(2.0, 3.0), 2)
            volume = round(length * width * height, 2)
            
            max_capacity = random.randint(1500, 2500)
            empty_weight = random.randint(1000, 1500)
            passenger_count = random.randint(1, 2)
            passenger_weight = passenger_count * random.randint(60, 90)
            cargo_weight = random.randint(0, 500)
            
            tire_pressure = round(random.uniform(40, 65), 1)
            maintenance_score = random.randint(1, 10)
            
            front_weight_percent = round(random.uniform(30, 40), 1)
            rear_weight_percent = round(100 - front_weight_percent, 1)
            
        elif vehicle_type == "heavy vehicle":
            length = round(random.uniform(7.0, 16.0), 2)
            width = round(random.uniform(2.5, 3.0), 2)
            height = round(random.uniform(3.0, 4.5), 2)
            volume = round(length * width * height, 2)
            
            max_capacity = random.randint(10000, 30000)
            empty_weight = random.randint(5000, 10000)
            passenger_count = random.randint(1, 3)
            passenger_weight = passenger_count * random.randint(60, 90)
            cargo_weight = random.randint(0, max_capacity)
            
            tire_pressure = round(random.uniform(80, 120), 1)
            maintenance_score = random.randint(1, 10)
            
            front_weight_percent = round(random.uniform(30, 40), 1)
            rear_weight_percent = round(100 - front_weight_percent, 1)

        # Calculate total vehicle weight
        weight = empty_weight + passenger_weight + cargo_weight
        
        # Volume utilization percentage
        if volume > 0:
            volume_utilization = min(100, round((cargo_weight / 200) / volume * 100, 1))  # Assuming average cargo density of 200 kg/m³
        else:
            volume_utilization = 0
            
        # Calculate load distribution in actual weight
        front_weight = round((weight * front_weight_percent) / 100, 1)
        rear_weight = round((weight * rear_weight_percent) / 100, 1)
        
        # Add some noise for realism (±2%)
        weight = int(weight * random.uniform(0.98, 1.02))
        
        # Fuel efficiency estimation (km/l) - decreases with load
        base_fuel_efficiency = {
            "2-wheeler": random.uniform(25, 40),
            "4-wheeler 5-seater": random.uniform(12, 18),
            "4-wheeler 7-seater": random.uniform(10, 16),
            "delivery vehicle": random.uniform(8, 14),
            "heavy vehicle": random.uniform(3, 8)
        }[vehicle_type]
        
        # Adjust for load (efficiency decreases with load)
        load_ratio = weight / max_capacity
        load_penalty = 1 - (load_ratio * 0.3)  # Up to 30% reduction at max load
        weather_penalty = 1.0
        if weather == "Rain":
            weather_penalty = 0.95
        elif weather == "Snow":
            weather_penalty = 0.85
        elif weather == "Fog":
            weather_penalty = 0.9
            
        road_penalty = {"Good": 1.0, "Fair": 0.9, "Poor": 0.8}[road_condition]
        
        fuel_efficiency = round(base_fuel_efficiency * load_penalty * weather_penalty * road_penalty, 2)
        
        # Determine overload status based on weight exceeding max capacity
        overload_status = "Overloaded" if weight > max_capacity else "Not Overloaded"
        
        # Calculate overload percentage
        if weight > max_capacity:
            overload_percentage = round(((weight - max_capacity) / max_capacity) * 100, 1)
        else:
            overload_percentage = 0.0
            
        # Risk factor (1-10 scale)
        base_risk = 3 if overload_status == "Not Overloaded" else 7
        road_risk_factor = {"Good": 0, "Fair": 1, "Poor": 3}[road_condition]
        weather_risk_factor = {"Clear": 0, "Rain": 1, "Snow": 3, "Fog": 2}[weather]
        maintenance_risk_factor = max(0, (5 - maintenance_score) / 2)  # Lower maintenance = higher risk
        
        risk_score = min(10, round(base_risk + road_risk_factor + weather_risk_factor + maintenance_risk_factor))
        
        # Append generated data to the list with all new features
        data.append([
            vehicle_type, weight, max_capacity, passenger_count, passenger_weight, cargo_weight, 
            length, width, height, volume, volume_utilization,
            front_weight_percent, rear_weight_percent, front_weight, rear_weight,
            region, road_condition, weather, temperature, is_winter, is_summer, hour_of_day, is_peak_hours,
            tire_pressure, maintenance_score, fuel_efficiency, risk_score, overload_percentage,
            overload_status
        ])

    # Convert generated data to a pandas DataFrame
    columns = [
        "vehicle_type", "weight", "max_load_capacity", "passenger_count", "passenger_weight", "cargo_weight",
        "length", "width", "height", "volume", "volume_utilization",
        "front_weight_percent", "rear_weight_percent", "front_weight", "rear_weight",
        "region", "road_condition", "weather", "temperature", "is_winter", "is_summer", "hour_of_day", "is_peak_hours",
        "tire_pressure", "maintenance_score", "fuel_efficiency", "risk_score", "overload_percentage",
        "overload_status"
    ]
    
    df = pd.DataFrame(data, columns=columns)

    # Add a small percentage of outliers for model robustness (about 1% of data)
    outlier_count = int(num_entries * 0.01)
    outlier_indices = random.sample(range(num_entries), outlier_count)
    
    for idx in outlier_indices:
        # Create extreme values for some features
        if random.choice([True, False]):
            # Extremely heavy load
            df.at[idx, 'cargo_weight'] = df.at[idx, 'max_load_capacity'] * random.uniform(1.5, 2.0)
            df.at[idx, 'weight'] = df.at[idx, 'weight'] - df.at[idx, 'cargo_weight'] + df.at[idx, 'cargo_weight']
            df.at[idx, 'overload_status'] = "Overloaded"
        else:
            # Unusual passenger count
            if df.at[idx, 'vehicle_type'] == "2-wheeler":
                df.at[idx, 'passenger_count'] = 3  # Too many passengers for a 2-wheeler
            else:
                max_pass = {"4-wheeler 5-seater": 7, "4-wheeler 7-seater": 9, "delivery vehicle": 5, "heavy vehicle": 5}
                df.at[idx, 'passenger_count'] = max_pass.get(df.at[idx, 'vehicle_type'], 5)

    # Perform initial data quality checks
    print("NaNs in df (original):", df.isnull().any().any())
    print("NaNs per column in df (original):\n", df.isnull().sum())
    print("Data types in df (original):\n", df.dtypes)

    # Define categorical and numerical features
    categorical_features = ["vehicle_type", "region", "road_condition", "weather"]
    numerical_features = [col for col in df.columns if col not in categorical_features + ["overload_status"]]
    
    # Separate features (X) and target variable (y)
    X = df.drop("overload_status", axis=1)
    y = df["overload_status"]

    # Perform one-hot encoding on categorical columns
    X = pd.get_dummies(X, columns=categorical_features)

    # Try both standard scaling and robust scaling, then compare
    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()
    
    X_numerical = X[numerical_features]
    
    X_scaled_standard = standard_scaler.fit_transform(X_numerical)
    X_scaled_robust = robust_scaler.fit_transform(X_numerical)
    
    # Create new DataFrames with scaled features
    X_scaled_standard_df = pd.DataFrame(X_scaled_standard, 
                                        columns=[f"{col}_standard_scaled" for col in numerical_features], 
                                        index=X.index)
    X_scaled_robust_df = pd.DataFrame(X_scaled_robust, 
                                        columns=[f"{col}_robust_scaled" for col in numerical_features], 
                                        index=X.index)

    # Apply multiple resampling techniques for comparison
    # 1. SMOTE
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)
    
    # 2. ADASYN (Adaptive Synthetic Sampling)
    adasyn = ADASYN(random_state=42)
    X_resampled_adasyn, y_resampled_adasyn = adasyn.fit_resample(X, y)
    
    # Choose SMOTE for final dataset (better for our case with clear class boundaries)
    X_resampled, y_resampled = X_resampled_smote, y_resampled_smote
    
    # Concatenate original and standard scaled features (more common)
    X_resampled = pd.concat([
        X_resampled, 
        pd.DataFrame(standard_scaler.transform(X_resampled[numerical_features]), 
                     columns=[f"{col}_scaled" for col in numerical_features], 
                     index=X_resampled.index)
    ], axis=1)

    # Combine resampled features and target variable
    df_balanced = pd.concat([X_resampled, y_resampled], axis=1)

    # Randomly sample to ensure manageable dataset size
    df_balanced = df_balanced.sample(n=20000, random_state=42, replace=False)

    # Create train and validation splits
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
    
    # Save each part separately
    train_df.to_csv("vehicle_data_train.csv", index=False)
    val_df.to_csv("vehicle_data_val.csv", index=False)

    # Perform final data quality checks before saving
    print("NaNs in df_balanced before saving:", df_balanced.isnull().any().any())
    print("NaNs per column in df_balanced before saving:\n", df_balanced.isnull().sum())
    print("Data types in df_balanced before saving:\n", df_balanced.dtypes)

    # Save the balanced and standardized dataset to a CSV file
    df_balanced.to_csv(dataset_filename, index=False)
    
    # Create metadata for documentation
    metadata = {
        "dataset_name": dataset_filename,
        "version": "2.0",
        "created_at": generation_timestamp,
        "num_samples": len(df_balanced),
        "features": {
            "categorical": categorical_features,
            "numerical": numerical_features,
            "target": "overload_status"
        },
        "preprocessing": {
            "scaling_methods": ["StandardScaler", "RobustScaler"],
            "resampling_methods": ["SMOTE", "ADASYN"],
            "final_chosen_methods": {
                "scaling": "StandardScaler",
                "resampling": "SMOTE"
            }
        },
        "feature_descriptions": {
            "vehicle_type": "Type of vehicle (categorical)",
            "weight": "Total weight of vehicle including passengers and cargo (kg)",
            "max_load_capacity": "Maximum load capacity of the vehicle (kg)",
            "passenger_count": "Number of passengers in the vehicle",
            "passenger_weight": "Total weight of passengers (kg)",
            "cargo_weight": "Weight of cargo being carried (kg)",
            "length": "Vehicle length in meters",
            "width": "Vehicle width in meters",
            "height": "Vehicle height in meters",
            "volume": "Vehicle cargo volume in cubic meters",
            "volume_utilization": "Percentage of volume utilized by cargo",
            "front_weight_percent": "Percentage of weight on front axle/wheels",
            "rear_weight_percent": "Percentage of weight on rear axle/wheels",
            "front_weight": "Actual weight on front axle/wheels (kg)",
            "rear_weight": "Actual weight on rear axle/wheels (kg)",
            "region": "Geographical region (Urban/Rural/Highway/Mountain)",
            "road_condition": "Condition of the road (Good/Fair/Poor)",
            "weather": "Weather condition (Clear/Rain/Snow/Fog)",
            "temperature": "Ambient temperature in Celsius",
            "is_winter": "Binary indicator if month is winter (Dec-Feb)",
            "is_summer": "Binary indicator if month is summer (Jun-Aug)",
            "hour_of_day": "Hour of the day (0-23)",
            "is_peak_hours": "Binary indicator if time is during peak hours",
            "tire_pressure": "Tire pressure in PSI",
            "maintenance_score": "Vehicle maintenance score (1-10, 10 being best)",
            "fuel_efficiency": "Estimated fuel efficiency in km/l",
            "risk_score": "Computed risk score (1-10, 10 being highest risk)",
            "overload_percentage": "Percentage by which vehicle is overloaded (0 if not overloaded)",
            "overload_status": "Target variable: Overloaded or Not Overloaded"
        },
        "class_distribution": {
            "Overloaded": y_resampled.value_counts()["Overloaded"],
            "Not Overloaded": y_resampled.value_counts()["Not Overloaded"],
        }
    }
    
    # Save metadata
    with open(metadata_filename, 'w') as file:
        yaml.dump(metadata, file, default_flow_style=False)
    
    print(f"Dataset metadata saved to '{metadata_filename}'.")
    print(f"Balanced and standardized dataset '{dataset_filename}' created with {len(df_balanced)} rows.")
    print(f"Train dataset saved to 'vehicle_data_train.csv' with {len(train_df)} rows.")
    print(f"Validation dataset saved to 'vehicle_data_val.csv' with {len(val_df)} rows.")

else:
    # If dataset already exists, print a message and skip generation
    print(f"Dataset '{dataset_filename}' already exists. Skipping data generation.")
    print("To regenerate the dataset, delete the existing file and run this script again.")