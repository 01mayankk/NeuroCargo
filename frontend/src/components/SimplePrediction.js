import React, { useState } from 'react';
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
  MenuItem
} from '@mui/material';
import {
  DirectionsCar,
  People,
  LocalShipping,
  Scale,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import { directApi } from '../services/api';

const SimplePrediction = () => {
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState({
    vehicle_type: '2-wheeler',
    weight: 100,
    max_load_capacity: 150,
    passenger_count: 1,
    cargo_weight: 20,
    weather: 'normal'
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState('');

  // Vehicle types
  const vehicleTypes = [
    '2-wheeler',
    '4-wheeler 5-seater',
    '4-wheeler 7-seater',
    'delivery vehicle',
    'heavy vehicle'
  ];

  // Weather options
  const weatherOptions = [
    { value: 'normal', label: 'Normal' },
    { value: 'rainy', label: 'Rainy' },
    { value: 'snowy', label: 'Snowy' },
    { value: 'windy', label: 'Windy' },
    { value: 'stormy', label: 'Stormy' }
  ];

  // Handle input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPredictionData({
      ...predictionData,
      [name]: ['passenger_count', 'weight', 'max_load_capacity', 'cargo_weight'].includes(name) 
        ? parseFloat(value) 
        : value
    });
  };

  // Handle prediction
  const handlePrediction = async () => {
    try {
      setLoading(true);
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
      const response = await directApi.predict(predictionData);
      
      if (response.data.status === 'success') {
        setPredictionResult(response.data);
        console.log(response.data);
      } else {
        setError('Prediction failed. Please check your inputs and try again.');
      }
    } catch (err) {
      console.error('Error making prediction:', err);
      setError('Error making prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Typography variant="h4" component="h1" sx={{ mb: 3 }}>
        Vehicle Load Prediction
      </Typography>

      {/* Prediction Form */}
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Enter Vehicle and Load Details
            </Typography>
            
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}
            
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <FormControl fullWidth size="small">
                  <InputLabel>Vehicle Type</InputLabel>
                  <Select
                    name="vehicle_type"
                    value={predictionData.vehicle_type}
                    onChange={handleInputChange}
                    label="Vehicle Type"
                  >
                    {vehicleTypes.map(type => (
                      <MenuItem key={type} value={type}>
                        {type}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Vehicle Weight (kg)"
                  name="weight"
                  type="number"
                  InputProps={{ inputProps: { min: 0 } }}
                  value={predictionData.weight}
                  onChange={handleInputChange}
                  variant="outlined"
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Max Load Capacity (kg)"
                  name="max_load_capacity"
                  type="number"
                  InputProps={{ inputProps: { min: 0 } }}
                  value={predictionData.max_load_capacity}
                  onChange={handleInputChange}
                  variant="outlined"
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Passenger Count"
                  name="passenger_count"
                  type="number"
                  InputProps={{ inputProps: { min: 0 } }}
                  value={predictionData.passenger_count}
                  onChange={handleInputChange}
                  variant="outlined"
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Cargo Weight (kg)"
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
                    onChange={handleInputChange}
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
                  disabled={loading}
                  sx={{ mt: 1 }}
                >
                  {loading ? (
                    <CircularProgress size={24} color="inherit" />
                  ) : (
                    'Make Prediction'
                  )}
                </Button>
              </Grid>
            </Grid>
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
                  bgcolor: predictionResult.prediction === 1 ? 'error.light' : 'success.light'
                }}
              >
                {predictionResult.prediction === 1 ? (
                  <Warning fontSize="large" color="error" sx={{ mr: 2 }} />
                ) : (
                  <CheckCircle fontSize="large" color="success" sx={{ mr: 2 }} />
                )}
                <Box>
                  <Typography variant="h5" color={predictionResult.prediction === 1 ? 'error.dark' : 'success.dark'}>
                    {predictionResult.prediction === 1 ? 'Vehicle Overloaded' : 'Vehicle Load is Safe'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {Math.round(predictionResult.confidence * 100)}%
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
                  {predictionResult.prediction === 1
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
                Enter vehicle and load details and make a prediction
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

export default SimplePrediction; 