// Vehicle Load Management Application - Frontend Script
// Provides enhanced functionality for the prediction interface

document.addEventListener('DOMContentLoaded', function() {
    // Hide welcome screen after animation
    setTimeout(function() {
        const welcomeScreen = document.querySelector('.welcome-screen');
        if (welcomeScreen) {
            welcomeScreen.classList.add('fade-out');
            
            // Remove it from DOM after fade out animation completes
            setTimeout(function() {
                welcomeScreen.style.display = 'none';
            }, 1000);
        }
    }, 4000);
    
    // Add ripple effect to buttons
    const rippleButtons = document.querySelectorAll('.ripple-button');
    rippleButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const rippleContainer = this.querySelector('.ripple-container');
            if (!rippleContainer) return;
            
            const rect = rippleContainer.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            const ripple = document.createElement('span');
            ripple.className = 'ripple';
            ripple.style.width = ripple.style.height = `${size}px`;
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            
            rippleContainer.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Form validation and highlighting
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value === '') {
                this.classList.remove('highlight-valid');
                this.classList.remove('highlight-invalid');
                return;
            }
            
            const value = parseFloat(this.value);
            if (isNaN(value) || value < 0) {
                this.classList.add('highlight-invalid');
                this.classList.remove('highlight-valid');
            } else {
                this.classList.add('highlight-valid');
                this.classList.remove('highlight-invalid');
            }
        });
    });
    
    // Image modal functionality
    const graphImages = document.querySelectorAll('.graph-image');
    const modal = document.getElementById('graphModal');
    const modalImg = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    const closeModal = document.querySelector('.modal-close');
    
    graphImages.forEach(img => {
        img.addEventListener('click', function() {
            modal.style.display = 'block';
            modalImg.src = this.src;
            modalCaption.textContent = this.alt;
        });
    });
    
    if (closeModal) {
        closeModal.addEventListener('click', function() {
            modal.style.display = 'none';
        });
    }
    
    // Close modal when clicking outside of it
    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // Handle form submission for vehicle load prediction
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitPredictionForm();
        });
    }
    
    // Handle try again button
    const tryAgainButton = document.querySelector('.try-again-button');
    if (tryAgainButton) {
        tryAgainButton.addEventListener('click', function() {
            document.getElementById('errorMessage').style.display = 'none';
            document.getElementById('predictionForm').reset();
            enableFormInputs();
        });
    }
    
    // Update UI for graphs by default
    updateGraphContainers();
});

function submitPredictionForm() {
    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const formObject = {};
    
    let hasInvalidInputs = false;
    
    // Convert form data to object and validate
    formData.forEach((value, key) => {
        if (key === 'weight' || key === 'max_load_capacity' || key === 'passenger_count' || key === 'cargo_weight') {
            const numValue = parseFloat(value);
            if (isNaN(numValue) || numValue < 0) {
                hasInvalidInputs = true;
                document.querySelector(`[name="${key}"]`).classList.add('highlight-invalid');
            }
            formObject[key] = numValue;
        } else {
            formObject[key] = value;
        }
    });
    
    // Stop submission if validation fails
    if (hasInvalidInputs) {
        document.getElementById('predictionForm').classList.add('shake');
        setTimeout(() => {
            document.getElementById('predictionForm').classList.remove('shake');
        }, 500);
        return;
    }
    
    // Show loading state
    document.getElementById('loadingResults').style.display = 'flex';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('predictionResults').style.display = 'none';
    
    // Disable form inputs while processing
    disableFormInputs();
    
    // Update graph loading indicators
    const graphContainers = document.querySelectorAll('.graph-container');
    graphContainers.forEach(container => {
        const loadingElement = container.querySelector('.graph-loading');
        if (loadingElement) {
            loadingElement.style.display = 'flex';
        }
    });
    
    // Make AJAX request to predict endpoint
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formObject)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Hide loading, show results
        document.getElementById('loadingResults').style.display = 'none';
        
        if (data.status === 'success') {
            displayPredictionResults(data);
            // Pass the single graph_url instead of expecting a graphs object
            if (data.graph_url) {
                const imgElement = document.getElementById('weightDistributionGraph');
                if (imgElement) {
                    // Add timestamp to prevent caching
                    imgElement.src = data.graph_url + '?t=' + new Date().getTime();
                    imgElement.alt = 'Weight distribution graph';
                    imgElement.style.display = 'block';
                }
            }
        } else {
            showError('Prediction Error', data.message || 'An unknown error occurred during prediction.');
        }
        
        enableFormInputs();
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('loadingResults').style.display = 'none';
        showError('Request Failed', 'Failed to connect to the prediction service. Please try again later.');
        enableFormInputs();
    });
}

function displayPredictionResults(data) {
    const resultsContainer = document.getElementById('predictionResults');
    resultsContainer.style.display = 'block';
    
    // Update prediction status
    const statusIndicator = document.querySelector('.status-indicator');
    const predictionValue = document.getElementById('predictionValue');
    
    if (data.prediction === 0 || data.prediction === "Not Overloaded") {
        statusIndicator.className = 'status-indicator safe';
        predictionValue.textContent = 'Not Overloaded';
    } else {
        statusIndicator.className = 'status-indicator danger';
        predictionValue.textContent = 'Overloaded';
    }
    
    // Update confidence level
    const confidenceValue = document.getElementById('confidenceValue');
    const gaugeFill = document.querySelector('.gauge-fill');
    
    if (data.confidence) {
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;
        gaugeFill.style.width = `${confidencePercent}%`;
    }
    
    // Update metrics
    if (data.metrics) {
        document.getElementById('loadPercentageValue').textContent = `${data.metrics.load_percentage}%`;
        document.getElementById('remainingCapacityValue').textContent = `${data.metrics.remaining_capacity} kg`;
        
        // Set risk level based on risk assessment
        const riskValue = document.getElementById('riskValue');
        const riskAssessment = data.metrics.risk_assessment;
        
        riskValue.textContent = riskAssessment;
        
        if (riskAssessment === 'Low') {
            riskValue.style.color = 'var(--success-color)';
        } else if (riskAssessment === 'Medium') {
            riskValue.style.color = 'var(--warning-color)';
        } else {
            riskValue.style.color = 'var(--danger-color)';
        }
    }
    
    // Animate the metrics appearance
    const metricItems = document.querySelectorAll('.metric-item');
    metricItems.forEach((item, index) => {
        item.style.opacity = '0';
        setTimeout(() => {
            item.style.opacity = '1';
            item.classList.add('fadeInUp');
        }, 100 * index);
    });
}

function updateGraphContainers() {
    // Generate graph containers with loading indicators by default
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            "vehicle_type": "2-wheeler",
            "weight": 1000,
            "max_load_capacity": 1500,
            "passenger_count": 3,
            "cargo_weight": 200,
            "region": "Urban",
            "road_condition": "Good",
            "weather": "Clear",
            "graph_refresh_only": true
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success' && data.graph_url) {
            const imgElement = document.getElementById('weightDistributionGraph');
            if (imgElement) {
                // Add timestamp to prevent caching
                imgElement.src = data.graph_url + '?t=' + new Date().getTime();
                imgElement.alt = 'Weight distribution graph';
                imgElement.style.display = 'block';
            }
        }
    })
    .catch(error => {
        console.error('Error loading initial graphs:', error);
    });
}

function showError(title, message) {
    const errorContainer = document.getElementById('errorMessage');
    const errorTitle = document.querySelector('.error-title');
    const errorDesc = document.querySelector('.error-description');
    
    errorTitle.textContent = title;
    errorDesc.textContent = message;
    
    errorContainer.style.display = 'block';
}

function disableFormInputs() {
    const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select, #predictionForm button');
    inputs.forEach(input => {
        input.disabled = true;
    });
}

function enableFormInputs() {
    const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select, #predictionForm button');
    inputs.forEach(input => {
        input.disabled = false;
    });
}