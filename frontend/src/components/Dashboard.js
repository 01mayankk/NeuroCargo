import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Button,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  Paper,
  Alert,
  Chip,
  Stack,
  IconButton,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Tabs,
  Tab
} from '@mui/material';
import {
  DirectionsCar,
  People,
  LocalShipping,
  Scale,
  Warning,
  CheckCircle,
  Timeline,
  BarChart,
  PieChart,
  Refresh,
  Add
} from '@mui/icons-material';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import Chart from 'react-apexcharts';

const Dashboard = () => {
  const navigate = useNavigate();
  const { getAuthHeaders } = useAuth();
  
  const [loading, setLoading] = useState(true);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [trendsData, setTrendsData] = useState(null);
  const [vehicles, setVehicles] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState('');
  const [timeRange, setTimeRange] = useState(30); // Default 30 days
  const [trendsTab, setTrendsTab] = useState(0); // 0: daily, 1: weekly, 2: monthly
  
  // Fetch dashboard data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError('');
        
        // Fetch analytics
        const analyticsResponse = await axios.get(
          `${process.env.REACT_APP_API_URL}/api/predictions/analytics`,
          { headers: getAuthHeaders() }
        );
        
        // Fetch trends
        const trendsResponse = await axios.get(
          `${process.env.REACT_APP_API_URL}/api/predictions/trends?days=${timeRange}`,
          { headers: getAuthHeaders() }
        );
        
        // Fetch vehicles
        const vehiclesResponse = await axios.get(
          `${process.env.REACT_APP_API_URL}/api/vehicles`,
          { headers: getAuthHeaders() }
        );
        
        // Fetch recent predictions
        const predictionsResponse = await axios.get(
          `${process.env.REACT_APP_API_URL}/api/predictions`,
          { headers: getAuthHeaders() }
        );
        
        // Set data from responses
        if (analyticsResponse.data.status === 'success') {
          setAnalyticsData(analyticsResponse.data.data);
        }
        
        if (trendsResponse.data.status === 'success') {
          setTrendsData(trendsResponse.data.data);
        }
        
        if (vehiclesResponse.data.status === 'success') {
          setVehicles(vehiclesResponse.data.data);
        }
        
        if (predictionsResponse.data.status === 'success') {
          setPredictions(predictionsResponse.data.data.slice(0, 10)); // Get latest 10
        }
        
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [getAuthHeaders, timeRange]);
  
  // Handle time range change
  const handleTimeRangeChange = (e) => {
    setTimeRange(parseInt(e.target.value));
  };
  
  // Handle refresh
  const handleRefresh = () => {
    setLoading(true);
    // Re-fetch data (useEffect will run again due to dependency on loading)
    setLoading(false);
  };
  
  // Handle trends tab change
  const handleTrendsTabChange = (event, newValue) => {
    setTrendsTab(newValue);
  };
  
  // Render loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  // Configuration for the load percentage chart
  const loadPercentageChartConfig = {
    options: {
      chart: {
        type: 'donut',
      },
      labels: ['Low Risk', 'Medium Risk', 'High Risk'],
      colors: ['#4caf50', '#ff9800', '#f44336'],
      legend: {
        position: 'bottom'
      },
      plotOptions: {
        pie: {
          donut: {
            labels: {
              show: true,
              total: {
                show: true,
                label: 'Total Predictions',
                fontSize: '14px',
                fontWeight: 600,
              }
            }
          }
        }
      }
    },
    series: analyticsData ? [
      analyticsData.risk_assessment.low,
      analyticsData.risk_assessment.medium,
      analyticsData.risk_assessment.high
    ] : [0, 0, 0]
  };
  
  // Configuration for the trends chart
  const getTrendsChartConfig = () => {
    let data = [];
    let categories = [];
    
    if (trendsData) {
      const selectedTabData = 
        trendsTab === 0 ? trendsData.daily : 
        trendsTab === 1 ? trendsData.weekly : 
        trendsData.monthly;
      
      categories = selectedTabData.map(item => 
        trendsTab === 0 ? item.date :
        trendsTab === 1 ? item.week :
        item.month
      );
      
      const overloadedData = selectedTabData.map(item => item.overloaded);
      const safeData = selectedTabData.map(item => item.safe);
      const avgLoadData = selectedTabData.map(item => item.avg_load_percentage);
      
      data = [
        {
          name: 'Overloaded',
          type: 'column',
          data: overloadedData
        },
        {
          name: 'Safe',
          type: 'column',
          data: safeData
        },
        {
          name: 'Avg Load %',
          type: 'line',
          data: avgLoadData
        }
      ];
    }
    
    return {
      options: {
        chart: {
          type: 'line',
          height: 350,
          stacked: false,
          toolbar: {
            show: true
          }
        },
        plotOptions: {
          bar: {
            columnWidth: '60%',
          }
        },
        stroke: {
          width: [0, 0, 3]
        },
        title: {
          text: `Load Prediction Trends ${
            trendsTab === 0 ? '(Daily)' : 
            trendsTab === 1 ? '(Weekly)' : 
            '(Monthly)'
          }`,
          align: 'left'
        },
        dataLabels: {
          enabled: false
        },
        labels: categories,
        xaxis: {
          type: 'category',
          categories: categories
        },
        yaxis: [
          {
            axisTicks: {
              show: true,
            },
            axisBorder: {
              show: true,
              color: '#008FFB'
            },
            labels: {
              style: {
                colors: '#008FFB',
              }
            },
            title: {
              text: "Count",
              style: {
                color: '#008FFB',
              }
            }
          },
          {
            opposite: true,
            axisTicks: {
              show: true,
            },
            axisBorder: {
              show: true,
              color: '#FEB019'
            },
            labels: {
              style: {
                colors: '#FEB019',
              }
            },
            title: {
              text: "Load Percentage",
              style: {
                color: '#FEB019',
              }
            }
          },
        ],
        tooltip: {
          fixed: {
            enabled: true,
            position: 'topLeft',
            offsetY: 30,
            offsetX: 60
          },
        },
        legend: {
          position: 'bottom',
        }
      },
      series: data
    };
  };
  
  // Configuration for vehicle stats chart
  const getVehicleStatsChartConfig = () => {
    if (!analyticsData || !analyticsData.vehicle_stats || analyticsData.vehicle_stats.length === 0) {
      return {
        options: {},
        series: []
      };
    }
    
    return {
      options: {
        chart: {
          type: 'bar',
          height: 350
        },
        plotOptions: {
          bar: {
            borderRadius: 4,
            horizontal: true,
          }
        },
        dataLabels: {
          enabled: false
        },
        colors: ['#4caf50', '#f44336'],
        stroke: {
          width: 1,
          colors: ['#fff']
        },
        xaxis: {
          categories: analyticsData.vehicle_stats.map(stat => stat.vehicle_name),
          title: {
            text: 'Count'
          }
        },
        yaxis: {
          title: {
            text: 'Vehicle'
          }
        },
        tooltip: {
          y: {
            formatter: function (val) {
              return val + " predictions"
            }
          }
        },
        legend: {
          position: 'bottom'
        }
      },
      series: [
        {
          name: 'Safe',
          data: analyticsData.vehicle_stats.map(stat => stat.prediction_count - stat.overloaded_count)
        },
        {
          name: 'Overloaded',
          data: analyticsData.vehicle_stats.map(stat => stat.overloaded_count)
        }
      ]
    };
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Dashboard
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => navigate('/vehicles/add')}
          >
            Add Vehicle
          </Button>
        </Box>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Vehicles
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <DirectionsCar sx={{ mr: 1, color: 'primary.main', fontSize: 40 }} />
                <Typography variant="h4">
                  {vehicles.length}
                </Typography>
              </Box>
              <Button
                variant="text"
                size="small"
                onClick={() => navigate('/vehicles')}
                sx={{ mt: 1 }}
              >
                View All
              </Button>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Total Predictions
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Timeline sx={{ mr: 1, color: 'info.main', fontSize: 40 }} />
                <Typography variant="h4">
                  {analyticsData ? analyticsData.total_predictions : 0}
                </Typography>
              </Box>
              {analyticsData && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Last prediction: {predictions.length > 0 ? new Date(predictions[0].created_at).toLocaleDateString() : 'N/A'}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Safe Predictions
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <CheckCircle sx={{ mr: 1, color: 'success.main', fontSize: 40 }} />
                <Typography variant="h4">
                  {analyticsData ? analyticsData.safe_count : 0}
                </Typography>
              </Box>
              {analyticsData && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {analyticsData.total_predictions > 0 
                    ? `${Math.round(100 - analyticsData.overloaded_percentage)}% of predictions`
                    : 'No predictions yet'
                  }
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>
                Overloaded Predictions
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Warning sx={{ mr: 1, color: 'error.main', fontSize: 40 }} />
                <Typography variant="h4">
                  {analyticsData ? analyticsData.overloaded_count : 0}
                </Typography>
              </Box>
              {analyticsData && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {analyticsData.total_predictions > 0 
                    ? `${Math.round(analyticsData.overloaded_percentage)}% of predictions`
                    : 'No predictions yet'
                  }
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Charts and Data */}
      <Grid container spacing={3}>
        {/* Risk Distribution Chart */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardHeader 
              title="Risk Assessment Distribution" 
              subheader={analyticsData?.total_predictions 
                ? `Based on ${analyticsData.total_predictions} predictions` 
                : 'No data available'}
            />
            <CardContent>
              {analyticsData && analyticsData.total_predictions > 0 ? (
                <Chart
                  options={loadPercentageChartConfig.options}
                  series={loadPercentageChartConfig.series}
                  type="donut"
                  height={350}
                />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 350 }}>
                  <Typography variant="body1" color="text.secondary">
                    No prediction data available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Vehicle Performance */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardHeader 
              title="Vehicle Performance" 
              subheader="Prediction results by vehicle"
            />
            <CardContent>
              {analyticsData && analyticsData.vehicle_stats.length > 0 ? (
                <Chart
                  options={getVehicleStatsChartConfig().options}
                  series={getVehicleStatsChartConfig().series}
                  type="bar"
                  height={350}
                />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 350 }}>
                  <Typography variant="body1" color="text.secondary">
                    No vehicle performance data available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Trends Chart */}
        <Grid item xs={12}>
          <Card>
            <CardHeader 
              title="Prediction Trends" 
              action={
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Time Range</InputLabel>
                  <Select
                    value={timeRange}
                    label="Time Range"
                    onChange={handleTimeRangeChange}
                  >
                    <MenuItem value={7}>Last 7 days</MenuItem>
                    <MenuItem value={30}>Last 30 days</MenuItem>
                    <MenuItem value={90}>Last 3 months</MenuItem>
                    <MenuItem value={180}>Last 6 months</MenuItem>
                  </Select>
                </FormControl>
              }
            />
            <Tabs 
              value={trendsTab} 
              onChange={handleTrendsTabChange}
              centered
              sx={{ mb: 2 }}
            >
              <Tab label="Daily" />
              <Tab label="Weekly" />
              <Tab label="Monthly" />
            </Tabs>
            <CardContent>
              {trendsData && ((trendsTab === 0 && trendsData.daily.length > 0) || 
                             (trendsTab === 1 && trendsData.weekly.length > 0) || 
                             (trendsTab === 2 && trendsData.monthly.length > 0)) ? (
                <Chart
                  options={getTrendsChartConfig().options}
                  series={getTrendsChartConfig().series}
                  type="line"
                  height={350}
                />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 350 }}>
                  <Typography variant="body1" color="text.secondary">
                    No trend data available for the selected period
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Recent Predictions */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardHeader 
              title="Recent Predictions" 
              action={
                <Button size="small" onClick={() => navigate('/predictions')}>
                  View All
                </Button>
              }
            />
            <CardContent>
              {predictions.length > 0 ? (
                <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                  {predictions.map((prediction, index) => (
                    <React.Fragment key={prediction.id}>
                      <ListItem
                        secondaryAction={
                          <Chip
                            label={prediction.is_overloaded ? 'Overloaded' : 'Safe'}
                            color={prediction.is_overloaded ? 'error' : 'success'}
                            size="small"
                          />
                        }
                      >
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              {prediction.is_overloaded ? (
                                <Warning color="error" sx={{ mr: 1, fontSize: 18 }} />
                              ) : (
                                <CheckCircle color="success" sx={{ mr: 1, fontSize: 18 }} />
                              )}
                              <Typography variant="body1">
                                {vehicles.find(v => v.id === prediction.vehicle_id)?.name || `Vehicle #${prediction.vehicle_id}`}
                              </Typography>
                            </Box>
                          }
                          secondary={
                            <Box sx={{ mt: 0.5 }}>
                              <Typography variant="body2" color="text.secondary">
                                {new Date(prediction.created_at).toLocaleString()}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Load: {prediction.load_percentage}% | Risk: {prediction.risk_assessment}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < predictions.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
                  <Typography variant="body1" color="text.secondary">
                    No prediction data available
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Vehicle Overview */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardHeader 
              title="Vehicle Overview" 
              action={
                <Button 
                  size="small" 
                  variant="contained" 
                  color="primary"
                  onClick={() => navigate('/vehicles/add')}
                >
                  Add New
                </Button>
              }
            />
            <CardContent>
              {vehicles.length > 0 ? (
                <List sx={{ maxHeight: 400, overflow: 'auto' }}>
                  {vehicles.map((vehicle, index) => (
                    <React.Fragment key={vehicle.id}>
                      <ListItem
                        secondaryAction={
                          <Button 
                            size="small" 
                            variant="outlined"
                            onClick={() => navigate(`/vehicles/${vehicle.id}/predict`)}
                          >
                            Predict
                          </Button>
                        }
                      >
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <DirectionsCar sx={{ mr: 1, fontSize: 18 }} />
                              <Typography variant="body1">
                                {vehicle.name}
                              </Typography>
                            </Box>
                          }
                          secondary={
                            <Box sx={{ mt: 0.5 }}>
                              <Typography variant="body2" color="text.secondary">
                                {vehicle.vehicle_type}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Max Capacity: {vehicle.max_load_capacity} kg
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < vehicles.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Box 
                  sx={{ 
                    display: 'flex', 
                    flexDirection: 'column',
                    justifyContent: 'center', 
                    alignItems: 'center', 
                    height: 200 
                  }}
                >
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                    No vehicles added yet
                  </Typography>
                  <Button 
                    variant="contained" 
                    startIcon={<Add />}
                    onClick={() => navigate('/vehicles/add')}
                  >
                    Add Your First Vehicle
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 