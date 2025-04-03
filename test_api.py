import requests
import json

def test_api():
    """Test basic API connectivity and endpoints"""
    base_url = 'http://127.0.0.1:5000'
    
    print("Testing API connectivity...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint response (status {response.status_code}):")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error connecting to root endpoint: {str(e)}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"\nHealth endpoint response (status {response.status_code}):")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error connecting to health endpoint: {str(e)}")
    
    # Test prediction endpoint with example data
    try:
        data = {
            "vehicle_type": "2-wheeler",
            "weight": 120,
            "max_load_capacity": 150,
            "passenger_count": 1,
            "cargo_weight": 20,
            "weather": "normal"
        }
        
        response = requests.post(f"{base_url}/predict", json=data)
        print(f"\nPrediction endpoint response (status {response.status_code}):")
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction result: {'Overloaded' if result.get('prediction') == 1 else 'Not Overloaded'}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
            print(f"Load percentage: {result.get('metrics', {}).get('load_percentage', 'N/A')}%")
        else:
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error connecting to prediction endpoint: {str(e)}")

if __name__ == "__main__":
    test_api() 