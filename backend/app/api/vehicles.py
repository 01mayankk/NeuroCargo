from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.vehicle import Vehicle, Prediction
from app.models.user import User
from app.ml.model import predict_load_status
from app import db
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Create blueprint
vehicles_bp = Blueprint('vehicles', __name__)

@vehicles_bp.route('/', methods=['GET'])
@jwt_required()
def get_vehicles():
    """Get all vehicles for the current user"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get user's vehicles
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
            
        # Return serialized vehicles
        return jsonify({
            'status': 'success',
            'data': [vehicle.to_dict() for vehicle in user.vehicles]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting vehicles: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': 'Error retrieving vehicles', 
            'error': str(e)
        }), 500

@vehicles_bp.route('/<int:vehicle_id>', methods=['GET'])
@jwt_required()
def get_vehicle(vehicle_id):
    """Get a specific vehicle by ID"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get specific vehicle
        vehicle = Vehicle.query.filter_by(id=vehicle_id, user_id=current_user_id).first()
        if not vehicle:
            return jsonify({'status': 'error', 'message': 'Vehicle not found'}), 404
            
        # Return serialized vehicle
        return jsonify({
            'status': 'success',
            'data': vehicle.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting vehicle {vehicle_id}: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': 'Error retrieving vehicle', 
            'error': str(e)
        }), 500

@vehicles_bp.route('/', methods=['POST'])
@jwt_required()
def create_vehicle():
    """Create a new vehicle"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'vehicle_type', 'weight', 'max_load_capacity']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error', 
                    'message': f'Missing required field: {field}'
                }), 400
                
        # Create new vehicle
        vehicle = Vehicle(
            user_id=current_user_id,
            name=data['name'],
            vehicle_type=data['vehicle_type'],
            weight=data['weight'],
            max_load_capacity=data['max_load_capacity'],
            registration_number=data.get('registration_number', ''),
            manufacturer=data.get('manufacturer', ''),
            model=data.get('model', ''),
            year=data.get('year'),
            color=data.get('color', '')
        )
        
        # Save to database
        db.session.add(vehicle)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Vehicle created successfully',
            'data': vehicle.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating vehicle: {str(e)}")
        db.session.rollback()
        return jsonify({
            'status': 'error', 
            'message': 'Error creating vehicle', 
            'error': str(e)
        }), 500

@vehicles_bp.route('/<int:vehicle_id>', methods=['PUT'])
@jwt_required()
def update_vehicle(vehicle_id):
    """Update a vehicle"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get vehicle
        vehicle = Vehicle.query.filter_by(id=vehicle_id, user_id=current_user_id).first()
        if not vehicle:
            return jsonify({'status': 'error', 'message': 'Vehicle not found'}), 404
            
        # Get JSON data
        data = request.get_json()
        
        # Update vehicle attributes
        if 'name' in data:
            vehicle.name = data['name']
        if 'vehicle_type' in data:
            vehicle.vehicle_type = data['vehicle_type']
        if 'weight' in data:
            vehicle.weight = data['weight']
        if 'max_load_capacity' in data:
            vehicle.max_load_capacity = data['max_load_capacity']
        if 'registration_number' in data:
            vehicle.registration_number = data['registration_number']
        if 'manufacturer' in data:
            vehicle.manufacturer = data['manufacturer']
        if 'model' in data:
            vehicle.model = data['model']
        if 'year' in data:
            vehicle.year = data['year']
        if 'color' in data:
            vehicle.color = data['color']
            
        # Save changes
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Vehicle updated successfully',
            'data': vehicle.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating vehicle {vehicle_id}: {str(e)}")
        db.session.rollback()
        return jsonify({
            'status': 'error', 
            'message': 'Error updating vehicle', 
            'error': str(e)
        }), 500

@vehicles_bp.route('/<int:vehicle_id>', methods=['DELETE'])
@jwt_required()
def delete_vehicle(vehicle_id):
    """Delete a vehicle"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get vehicle
        vehicle = Vehicle.query.filter_by(id=vehicle_id, user_id=current_user_id).first()
        if not vehicle:
            return jsonify({'status': 'error', 'message': 'Vehicle not found'}), 404
            
        # Delete vehicle
        db.session.delete(vehicle)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Vehicle deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting vehicle {vehicle_id}: {str(e)}")
        db.session.rollback()
        return jsonify({
            'status': 'error', 
            'message': 'Error deleting vehicle', 
            'error': str(e)
        }), 500
        
@vehicles_bp.route('/<int:vehicle_id>/predict', methods=['POST'])
@jwt_required()
def predict_load(vehicle_id):
    """Make a load prediction for a vehicle"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get vehicle
        vehicle = Vehicle.query.filter_by(id=vehicle_id, user_id=current_user_id).first()
        if not vehicle:
            return jsonify({'status': 'error', 'message': 'Vehicle not found'}), 404
            
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['passenger_count', 'cargo_weight']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error', 
                    'message': f'Missing required field: {field}'
                }), 400
                
        # Combine vehicle data with request data
        prediction_data = {
            'vehicle_type': vehicle.vehicle_type,
            'weight': vehicle.weight,
            'max_load_capacity': vehicle.max_load_capacity,
            'passenger_count': data['passenger_count'],
            'cargo_weight': data['cargo_weight'],
            'weather': data.get('weather', 'normal')
        }
        
        # Get prediction from ML model
        prediction_result = predict_load_status(prediction_data)
        
        # Create prediction record
        prediction = Prediction(
            vehicle_id=vehicle_id,
            passenger_count=data['passenger_count'],
            cargo_weight=data['cargo_weight'],
            weather_condition=data.get('weather', 'normal'),
            is_overloaded=bool(prediction_result['prediction']),
            confidence=prediction_result['confidence'],
            load_percentage=prediction_result['metrics']['load_percentage'],
            risk_assessment=prediction_result['metrics']['risk_assessment']
        )
        
        # Save to database
        db.session.add(prediction)
        db.session.commit()
        
        # Return prediction results
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': prediction.to_dict(),
                'metrics': prediction_result['metrics']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error making prediction for vehicle {vehicle_id}: {str(e)}")
        db.session.rollback()
        return jsonify({
            'status': 'error', 
            'message': 'Error making prediction', 
            'error': str(e)
        }), 500
        
@vehicles_bp.route('/<int:vehicle_id>/predictions', methods=['GET'])
@jwt_required()
def get_predictions(vehicle_id):
    """Get all predictions for a vehicle"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get vehicle
        vehicle = Vehicle.query.filter_by(id=vehicle_id, user_id=current_user_id).first()
        if not vehicle:
            return jsonify({'status': 'error', 'message': 'Vehicle not found'}), 404
            
        # Get predictions
        predictions = Prediction.query.filter_by(vehicle_id=vehicle_id).order_by(Prediction.created_at.desc()).all()
        
        # Return serialized predictions
        return jsonify({
            'status': 'success',
            'data': [prediction.to_dict() for prediction in predictions]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting predictions for vehicle {vehicle_id}: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': 'Error retrieving predictions', 
            'error': str(e)
        }), 500 