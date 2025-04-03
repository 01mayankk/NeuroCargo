import axios from 'axios';

// Create base API instance
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to attach auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor to handle common errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Log error details for debugging
      console.error('API Error:', error.response.status, error.response.data);
      
      // Handle 401 Unauthorized
      if (error.response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
    } else if (error.request) {
      console.error('Network Error:', error.request);
    } else {
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// Direct API endpoints for simple model
export const directApi = {
  predict: (data) => api.post('/predict', data),
  health: () => api.get('/health'),
};

// Auth endpoints for the full app
export const authService = {
  login: (credentials) => api.post('/api/auth/login', credentials),
  register: (userData) => api.post('/api/auth/register', userData),
  getProfile: () => api.get('/api/auth/profile'),
  updateProfile: (userData) => api.put('/api/auth/profile', userData),
};

// Vehicle endpoints for the full app
export const vehicleService = {
  getAll: () => api.get('/api/vehicles'),
  getById: (id) => api.get(`/api/vehicles/${id}`),
  create: (vehicleData) => api.post('/api/vehicles', vehicleData),
  update: (id, vehicleData) => api.put(`/api/vehicles/${id}`, vehicleData),
  delete: (id) => api.delete(`/api/vehicles/${id}`),
  predict: (id, predictionData) => api.post(`/api/vehicles/${id}/predict`, predictionData),
  getPredictions: (id) => api.get(`/api/vehicles/${id}/predictions`),
};

// Prediction endpoints for the full app
export const predictionService = {
  getAll: () => api.get('/api/predictions'),
  getAnalytics: () => api.get('/api/predictions/analytics'),
  getTrends: (days = 30) => api.get(`/api/predictions/trends?days=${days}`),
};

export default api; 