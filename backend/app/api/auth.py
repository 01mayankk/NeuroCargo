from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import timedelta
from ..models.user import User
from .. import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json()
    
    # Check if all required fields are present
    if not data or not all(k in data for k in ('username', 'email', 'password')):
        return jsonify({
            'status': 'error',
            'message': 'Missing required fields'
        }), 400
    
    # Check if username already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({
            'status': 'error',
            'message': 'Username already exists'
        }), 409
    
    # Check if email already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({
            'status': 'error',
            'message': 'Email already exists'
        }), 409
    
    # Create new user
    try:
        user = User(
            username=data['username'],
            email=data['email'],
            password=data['password']
        )
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error registering user: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error registering user',
            'error': str(e)
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Log in a user and return a JWT token."""
    data = request.get_json()
    
    # Check if all required fields are present
    if not data or not all(k in data for k in ('username', 'password')):
        return jsonify({
            'status': 'error',
            'message': 'Missing username or password'
        }), 400
    
    # Find user by username
    user = User.query.filter_by(username=data['username']).first()
    
    # Check if user exists and password is correct
    if not user or not user.check_password(data['password']):
        return jsonify({
            'status': 'error',
            'message': 'Invalid username or password'
        }), 401
    
    # Create access token
    expires = timedelta(days=1)
    access_token = create_access_token(
        identity=user.id,
        expires_delta=expires
    )
    
    return jsonify({
        'status': 'success',
        'message': 'Login successful',
        'access_token': access_token,
        'user_id': user.id,
        'username': user.username
    }), 200

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get the profile of the authenticated user."""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': 'User not found'
        }), 404
    
    return jsonify({
        'status': 'success',
        'user': user.to_dict()
    }), 200

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update the profile of the authenticated user."""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({
            'status': 'error',
            'message': 'User not found'
        }), 404
    
    data = request.get_json()
    
    try:
        # Update username if provided and not already taken
        if 'username' in data and data['username'] != user.username:
            if User.query.filter_by(username=data['username']).first():
                return jsonify({
                    'status': 'error',
                    'message': 'Username already exists'
                }), 409
            user.username = data['username']
        
        # Update email if provided and not already taken
        if 'email' in data and data['email'] != user.email:
            if User.query.filter_by(email=data['email']).first():
                return jsonify({
                    'status': 'error',
                    'message': 'Email already exists'
                }), 409
            user.email = data['email']
        
        # Update password if provided
        if 'password' in data:
            user.set_password(data['password'])
        
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
    
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating profile: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error updating profile',
            'error': str(e)
        }), 500 