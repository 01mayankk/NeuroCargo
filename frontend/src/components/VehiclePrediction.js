import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Slider,
  Divider,
  Alert,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip
} from '@mui/material';
import {
  DirectionsCar,
  People,
  LocalShipping,
  Scale,
  Speed,
  Warning,
  CheckCircle,
  ArrowBack
} from '@mui/icons-material';
import { vehicleService } from '../services/api';
import { useAuth } from '../context/AuthContext';
import Chart from 'react-apexcharts';

const VehiclePrediction = () => {
  const { vehicleId } = useParams();
  const navigate = useNavigate();
  const { getAuthHeaders } = useAuth();
  
  const [loading, setLoading] = useState(false);
  const [loadingVehicle, setLoadingVehicle] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [vehicle, setVehicle] = useState(null);
  const [predictionData, setPredictionData] = useState({
    passenger_count: 0,
    cargo_weight: 0,
    weather: 'normal'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState('');
  const [recentPredictions, setRecentPredictions] = useState([]);

  // Weather options
  const weatherOptions = [
    { value: 'normal', label: 'Normal' },
    { value: 'rainy', label: 'Rainy' },
    { value: 'snowy', label: 'Snowy' },
    { value: 'windy', label: 'Windy' },
    { value: 'stormy', label: 'Stormy' }
  ];

  // Fetch vehicle details
  useEffect(() => {
    const fetchVehicle = async () => {
      try {
        setLoadingVehicle(true);
        const response = await vehicleService.getById(vehicleId);
        
        if (response.data.status === 'success') {
          setVehicle(response.data.data);
        } else {
          setError('Failed to load vehicle details');
        }
      } catch (err) {
        console.error('Error fetching vehicle:', err);
        setError('Error loading vehicle details. Please try again.');
      } finally {
        setLoadingVehicle(false);
      }
    };

    const fetchPredictions = async () => {
      try {
        setLoading(true);
        const response = await vehicleService.getPredictions(vehicleId);
        
        if (response.data.status === 'success') {
          // Get the 5 most recent predictions
          setRecentPredictions(response.data.data.slice(0, 5));
        }
      } catch (err) {
        console.error('Error fetching predictions:', err);
      } finally {
        setLoading(false);
      }
    };

    if (vehicleId) {
      fetchVehicle();
      fetchPredictions();
    }
  }, [vehicleId]);

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPredictionData({
      ...predictionData,
      [name]: name === 'passenger_count' ? parseInt(value) : parseFloat(value)
    });
  };

  // Handle weather change
  const handleWeatherChange = (e) => {
    setPredictionData({
      ...predictionData,
      weather: e.target.value
    });
  };

  // Handle prediction
  const handlePrediction = async () => {
    try {
      setPredicting(true);
      setError('');
      
      // Validate inputs
      if (predictionData.passenger_count < 0) {
        setError('Passenger count cannot be negative');
        return;
      }
      
      if (predictionData.cargo_weight < 0) {
        setError('Cargo weight cannot be negative');
        return;
      }
      
      // Make prediction API call
      const response = await vehicleService.predict(vehicleId, predictionData);
      
      if (response.data.status === 'success') {
        setPredictionResult(response.data.data);
        
        // Refresh predictions list
        const predictionsResponse = await vehicleService.getPredictions(vehicleId);
        
        if (predictionsResponse.data.status === 'success') {
          setRecentPredictions(predictionsResponse.data.data.slice(0, 5));
        }
      } else {
        setError('Prediction failed. Please check your inputs and try again.');
      }
    } catch (err) {
      console.error('Error making prediction:', err);
      setError('Error making prediction. Please try again.');
    } finally {
      setPredicting(false);
    }
  };

  // Render loading state
  if (loadingVehicle) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  // Render error state
  if (!vehicle) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          {error || 'Vehicle not found'}
        </Alert>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/vehicles')}
          sx={{ mt: 2 }}
        >
          Back to Vehicles
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/vehicles')}
          sx={{ mr: 2 }}
        >
          Back
        </Button>
        <Typography variant="h4" component="h1">
          Predict Load for {vehicle.name}
        </Typography>
      </Box>

      {/* Vehicle Info Card */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <DirectionsCar sx={{ mr: 1 }} />
                <Typography variant="h6">
                  {vehicle.name} ({vehicle.vehicle_type})
                </Typography>
              </Box>
              {vehicle.registration_number && (
                <Typography variant="body1" color="text.secondary">
                  Reg. No: {vehicle.registration_number}
                </Typography>
              )}
              {vehicle.manufacturer && (
                <Typography variant="body1" color="text.secondary">
                  {vehicle.manufacturer} {vehicle.model} {vehicle.year}
                </Typography>
              )}
            </Grid>
            <Grid item xs={12} sm={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Scale sx={{ mr: 1 }} />
                <Typography variant="body1">
                  Weight: {vehicle.weight} kg
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <LocalShipping sx={{ mr: 1 }} />
                <Typography variant="body1">
                  Max Load Capacity: {vehicle.max_load_capacity} kg
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Prediction Form */}
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Enter Load Details
            </Typography>
            
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <People sx={{ mr: 1 }} />
                  <Typography variant="body1">Passenger Count</Typography>
                </Box>
                <TextField
                  fullWidth
                  name="passenger_count"
                  type="number"
                  InputProps={{ inputProps: { min: 0 } }}
                  value={predictionData.passenger_count}
                  onChange={handleInputChange}
                  variant="outlined"
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <LocalShipping sx={{ mr: 1 }} />
                  <Typography variant="body1">Cargo Weight (kg)</Typography>
                </Box>
                <TextField
                  fullWidth
                  name="cargo_weight"
                  type="number"
                  InputProps={{ inputProps: { min: 0 } }}
                  value={predictionData.cargo_weight}
                  onChange={handleInputChange}
                  variant="outlined"
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControl fullWidth size="small">
                  <InputLabel>Weather Condition</InputLabel>
                  <Select
                    name="weather"
                    value={predictionData.weather}
                    onChange={handleWeatherChange}
                    label="Weather Condition"
                  >
                    {weatherOptions.map(option => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <Button
                  fullWidth
                  variant="contained"
                  color="primary"
                  onClick={handlePrediction}
                  disabled={predicting}
                  sx={{ mt: 1 }}
                >
                  {predicting ? (
                    <CircularProgress size={24} color="inherit" />
                  ) : (
                    'Make Prediction'
                  )}
                </Button>
              </Grid>
            </Grid>
          </Paper>
          
          {/* Recent Predictions */}
          <Paper sx={{ p: 3, mt: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Recent Predictions
            </Typography>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                <CircularProgress size={24} />
              </Box>
            ) : recentPredictions.length === 0 ? (
              <Typography variant="body2" color="text.secondary">
                No predictions have been made for this vehicle yet.
              </Typography>
            ) : (
              recentPredictions.map((prediction, index) => (
                <Box key={prediction.id} sx={{ mb: index === recentPredictions.length - 1 ? 0 : 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {prediction.is_overloaded ? (
                        <Warning color="error" sx={{ mr: 1 }} />
                      ) : (
                        <CheckCircle color="success" sx={{ mr: 1 }} />
                      )}
                      <Typography variant="body1">
                        {new Date(prediction.created_at).toLocaleString()}
                      </Typography>
                    </Box>
                    <Chip
                      label={prediction.is_overloaded ? 'Overloaded' : 'Safe'}
                      color={prediction.is_overloaded ? 'error' : 'success'}
                      size="small"
                    />
                  </Box>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                    <Typography variant="body2" color="text.secondary">
                      Passengers: {prediction.passenger_count}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Cargo: {prediction.cargo_weight} kg
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Load: {prediction.load_percentage}%
                    </Typography>
                  </Box>
                  
                  {index < recentPredictions.length - 1 && <Divider sx={{ mt: 2 }} />}
                </Box>
              ))
            )}
          </Paper>
        </Grid>
        
        {/* Prediction Results */}
        <Grid item xs={12} md={6}>
          {predictionResult ? (
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Prediction Results
              </Typography>
              
              {/* Prediction Outcome */}
              <Box 
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 3,
                  p: 2,
                  borderRadius: 1,
                  bgcolor: predictionResult.prediction.is_overloaded ? 'error.light' : 'success.light'
                }}
              >
                {predictionResult.prediction.is_overloaded ? (
                  <Warning fontSize="large" color="error" sx={{ mr: 2 }} />
                ) : (
                  <CheckCircle fontSize="large" color="success" sx={{ mr: 2 }} />
                )}
                <Box>
                  <Typography variant="h5" color={predictionResult.prediction.is_overloaded ? 'error.dark' : 'success.dark'}>
                    {predictionResult.prediction.is_overloaded ? 'Vehicle Overloaded' : 'Vehicle Load is Safe'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {Math.round(predictionResult.prediction.confidence * 100)}%
                  </Typography>
                </Box>
              </Box>
              
              {/* Load Percentage Meter */}
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body1">
                    Load Percentage
                  </Typography>
                  <Typography 
                    variant="body1" 
                    color={
                      predictionResult.metrics.load_percentage < 70 ? 'success.main' :
                      predictionResult.metrics.load_percentage < 90 ? 'warning.main' : 
                      'error.main'
                    }
                  >
                    {predictionResult.metrics.load_percentage}%
                  </Typography>
                </Box>
                <Box sx={{ px: 1 }}>
                  <Slider
                    value={predictionResult.metrics.load_percentage}
                    min={0}
                    max={100}
                    disabled
                    marks={[
                      { value: 0, label: '0%' },
                      { value: 50, label: '50%' },
                      { value: 100, label: '100%' },
                    ]}
                    sx={{
                      '& .MuiSlider-rail': {
                        background: 'linear-gradient(90deg, #4caf50 0%, #4caf50 70%, #ff9800 70%, #ff9800 90%, #f44336 90%, #f44336 100%)',
                        opacity: 1
                      }
                    }}
                  />
                </Box>
              </Box>
              
              {/* Risk Assessment */}
              <Box sx={{ mb: 3 }}>
                <Typography variant="body1" sx={{ mb: 1 }}>
                  Risk Assessment: <strong>{predictionResult.metrics.risk_assessment}</strong>
                </Typography>
                <Alert 
                  severity={
                    predictionResult.metrics.risk_assessment === 'Low' ? 'success' :
                    predictionResult.metrics.risk_assessment === 'Medium' ? 'warning' :
                    'error'
                  }
                  sx={{ mb: 2 }}
                >
                  {predictionResult.metrics.risk_assessment === 'Low' && 'Vehicle is loaded within safe limits.'}
                  {predictionResult.metrics.risk_assessment === 'Medium' && 'Vehicle is approaching maximum load capacity. Use caution.'}
                  {predictionResult.metrics.risk_assessment === 'High' && 'Vehicle is at or exceeding safe load capacity. Not recommended for travel.'}
                </Alert>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              {/* Additional Metrics */}
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Total Weight:
                  </Typography>
                  <Typography variant="body1">
                    {predictionResult.metrics.total_weight} kg
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Remaining Capacity:
                  </Typography>
                  <Typography variant="body1">
                    {predictionResult.metrics.remaining_capacity} kg
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Passenger Weight:
                  </Typography>
                  <Typography variant="body1">
                    {predictionResult.metrics.passenger_weight} kg
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Fuel Efficiency Impact:
                  </Typography>
                  <Typography variant="body1">
                    {predictionResult.metrics.fuel_efficiency_impact}% reduction
                  </Typography>
                </Grid>
              </Grid>
              
              {/* Recommendation */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="body1" sx={{ mb: 1 }}>
                  Recommendation:
                </Typography>
                <Typography variant="body2">
                  {predictionResult.prediction.is_overloaded 
                    ? "Reduce the load to ensure safe operation of the vehicle. Consider removing cargo or reducing passenger count."
                    : "The vehicle is loaded within acceptable parameters and is safe to operate."
                  }
                </Typography>
              </Box>
            </Paper>
          ) : (
            <Paper 
              sx={{ 
                p: 3, 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                minHeight: 300
              }}
            >
              <Scale sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" align="center">
                Enter load details and make a prediction
              </Typography>
              <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
                The ML model will analyze the data and determine if the vehicle is safely loaded
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default VehiclePrediction; 