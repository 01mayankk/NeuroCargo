# Vehicle Load Management System

A web application to assess vehicle load safety and provide recommendations based on vehicle specifications, passenger count, and cargo weight.

![Vehicle Load Management Screenshot](static/images/screenshot.png)

## Features

- **Load Status Prediction**: Determine if a vehicle is overloaded based on vehicle weight, passenger count, and cargo
- **Risk Assessment**: Get detailed risk analysis for vehicle loads
- **Interactive Dashboard**: Visual representation of load data with responsive UI
- **Automatic Suggestions**: Receive recommendations for safer load distribution
- **Visualizations**: View weight distribution through charts and graphs

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vehicle-load-management.git
   cd vehicle-load-management
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`

## Usage Guide

1. **Select Vehicle Type**
   - Choose from available vehicle types (2-wheeler, 4-wheeler, etc.)
   - Default specifications will be populated based on selection

2. **Enter Vehicle Details**
   - Vehicle Weight (kg)
   - Maximum Load Capacity (kg)
   - Number of Passengers
   - Cargo Weight (kg)

3. **Check Load Status**
   - Click "Check Load Status" to calculate and display results
   - View prediction result (Overloaded or Not Overloaded)
   - Review detailed metrics:
     - Load Percentage
     - Remaining Capacity
     - Risk Assessment
     - Total Weight
     - Fuel Efficiency Impact

4. **Interpret Results**
   - Green indicators: Safe load levels
   - Yellow indicators: Approaching maximum safe load
   - Red indicators: Unsafe load levels

## Technical Information

### System Architecture

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript (with Bootstrap 5)
- **ML Component**: Scikit-learn Random Forest model
- **Data Visualization**: Matplotlib

### Model Information

The application uses a machine learning model to predict load status. If no existing model is found, a placeholder model is created automatically. The model considers:

- Vehicle type (one-hot encoded)
- Vehicle weight (scaled)
- Maximum load capacity (scaled)
- Passenger count (scaled)
- Cargo weight (scaled)

### File Structure

```
vehicle-load-management/
├── app.py                  # Main Flask application
├── models/                 # Directory for ML models
│   ├── vehicle_load_model.pkl
│   └── vehicle_load_scaler.pkl
├── static/                 # Static files
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── weight_distribution.png
├── templates/              # HTML templates
│   └── index.html
└── README.md
```

## Customization

### Adding New Vehicle Types

To add a new vehicle type:

1. Update the vehicle type dropdown in `templates/index.html`
2. Add appropriate weight/load defaults in the JavaScript section
3. If using a trained model, ensure it supports the new vehicle type

### Extending Functionality

- **Custom Risk Models**: Modify the risk assessment logic in `calculate_metrics()` function
- **Additional Metrics**: Add new metrics to the returned dictionary in `calculate_metrics()`
- **Enhanced Visualizations**: Extend the `generate_graphs()` function to create more visualizations

## Troubleshooting

**Issue**: Graphs not displaying
- Ensure the `static/images` directory exists and is writable
- Check browser console for any JavaScript errors

**Issue**: Model prediction errors
- Verify input data is in the correct format
- Check if the model file exists and is valid
- Look at the application logs for specific errors

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Bootstrap team for the responsive UI framework
- Scikit-learn contributors for the machine learning tools
- Flask team for the web framework

---

For any questions or support, please open an issue on GitHub or contact the development team. 