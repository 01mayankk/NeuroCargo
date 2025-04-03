from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.vehicle import Vehicle, Prediction
from app.models.user import User
from app import db
from sqlalchemy import func
import logging
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger(__name__)

# Create blueprint
predictions_bp = Blueprint('predictions', __name__)

@predictions_bp.route('/', methods=['GET'])
@jwt_required()
def get_all_predictions():
    """Get all predictions for the current user across all vehicles"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get user's vehicles
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
            
        # Get vehicle IDs for the user
        vehicle_ids = [vehicle.id for vehicle in user.vehicles]
        
        # Return error if no vehicles exist
        if not vehicle_ids:
            return jsonify({
                'status': 'success',
                'data': []
            }), 200
            
        # Get predictions for all vehicles
        predictions = Prediction.query.filter(
            Prediction.vehicle_id.in_(vehicle_ids)
        ).order_by(Prediction.created_at.desc()).all()
        
        # Return serialized predictions
        return jsonify({
            'status': 'success',
            'data': [prediction.to_dict() for prediction in predictions]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting all predictions: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': 'Error retrieving predictions', 
            'error': str(e)
        }), 500

@predictions_bp.route('/analytics', methods=['GET'])
@jwt_required()
def get_prediction_analytics():
    """Get prediction analytics for the current user"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get user's vehicles
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
            
        # Get vehicle IDs for the user
        vehicle_ids = [vehicle.id for vehicle in user.vehicles]
        
        # Return error if no vehicles exist
        if not vehicle_ids:
            return jsonify({
                'status': 'success',
                'data': {
                    'total_predictions': 0,
                    'overloaded_count': 0,
                    'safe_count': 0,
                    'overloaded_percentage': 0,
                    'risk_assessment': {
                        'high': 0,
                        'medium': 0,
                        'low': 0
                    },
                    'vehicle_stats': []
                }
            }), 200
            
        # Get total predictions count
        total_predictions = Prediction.query.filter(
            Prediction.vehicle_id.in_(vehicle_ids)
        ).count()
        
        # Get overloaded predictions count
        overloaded_count = Prediction.query.filter(
            Prediction.vehicle_id.in_(vehicle_ids),
            Prediction.is_overloaded == True
        ).count()
        
        # Get safe predictions count
        safe_count = total_predictions - overloaded_count
        
        # Calculate overloaded percentage
        overloaded_percentage = 0
        if total_predictions > 0:
            overloaded_percentage = (overloaded_count / total_predictions) * 100
            
        # Get risk assessment counts
        high_risk_count = Prediction.query.filter(
            Prediction.vehicle_id.in_(vehicle_ids),
            Prediction.risk_assessment == 'High'
        ).count()
        
        medium_risk_count = Prediction.query.filter(
            Prediction.vehicle_id.in_(vehicle_ids),
            Prediction.risk_assessment == 'Medium'
        ).count()
        
        low_risk_count = Prediction.query.filter(
            Prediction.vehicle_id.in_(vehicle_ids),
            Prediction.risk_assessment == 'Low'
        ).count()
        
        # Get stats for each vehicle
        vehicle_stats = []
        for vehicle_id in vehicle_ids:
            vehicle = Vehicle.query.get(vehicle_id)
            
            # Get predictions for this vehicle
            vehicle_predictions = Prediction.query.filter_by(vehicle_id=vehicle_id).all()
            
            # Skip if no predictions
            if not vehicle_predictions:
                continue
                
            # Calculate stats
            vehicle_total = len(vehicle_predictions)
            vehicle_overloaded = sum(1 for p in vehicle_predictions if p.is_overloaded)
            vehicle_overloaded_percent = (vehicle_overloaded / vehicle_total * 100) if vehicle_total > 0 else 0
            
            # Calculate average load percentage
            avg_load_percentage = sum(p.load_percentage for p in vehicle_predictions) / vehicle_total if vehicle_total > 0 else 0
            
            # Get most recent prediction
            most_recent = max(vehicle_predictions, key=lambda p: p.created_at)
            
            vehicle_stats.append({
                'vehicle_id': vehicle.id,
                'vehicle_name': vehicle.name,
                'prediction_count': vehicle_total,
                'overloaded_count': vehicle_overloaded,
                'overloaded_percentage': round(vehicle_overloaded_percent, 2),
                'avg_load_percentage': round(avg_load_percentage, 2),
                'most_recent_prediction': most_recent.to_dict()
            })
            
        # Return analytics data
        return jsonify({
            'status': 'success',
            'data': {
                'total_predictions': total_predictions,
                'overloaded_count': overloaded_count,
                'safe_count': safe_count,
                'overloaded_percentage': round(overloaded_percentage, 2),
                'risk_assessment': {
                    'high': high_risk_count,
                    'medium': medium_risk_count,
                    'low': low_risk_count
                },
                'vehicle_stats': vehicle_stats
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting prediction analytics: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': 'Error retrieving prediction analytics', 
            'error': str(e)
        }), 500

@predictions_bp.route('/trends', methods=['GET'])
@jwt_required()
def get_prediction_trends():
    """Get prediction trends over time"""
    try:
        # Get current user ID from JWT
        current_user_id = get_jwt_identity()
        
        # Get user's vehicles
        user = User.query.get(current_user_id)
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
            
        # Get vehicle IDs for the user
        vehicle_ids = [vehicle.id for vehicle in user.vehicles]
        
        # Return empty data if no vehicles exist
        if not vehicle_ids:
            return jsonify({
                'status': 'success',
                'data': {
                    'daily': [],
                    'weekly': [],
                    'monthly': []
                }
            }), 200
            
        # Get period from query parameters (default: 30 days)
        days = request.args.get('days', 30, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        # Get predictions within time period
        predictions = Prediction.query.filter(
            Prediction.vehicle_id.in_(vehicle_ids),
            Prediction.created_at >= start_date
        ).all()
        
        # Generate daily data
        daily_data = {}
        for prediction in predictions:
            date_key = prediction.created_at.strftime('%Y-%m-%d')
            
            if date_key not in daily_data:
                daily_data[date_key] = {
                    'date': date_key,
                    'total': 0,
                    'overloaded': 0,
                    'safe': 0,
                    'avg_load_percentage': 0,
                    'total_load_percentage': 0
                }
                
            daily_data[date_key]['total'] += 1
            daily_data[date_key]['overloaded'] += 1 if prediction.is_overloaded else 0
            daily_data[date_key]['safe'] += 0 if prediction.is_overloaded else 1
            daily_data[date_key]['total_load_percentage'] += prediction.load_percentage
            
        # Calculate averages
        for date_key in daily_data:
            if daily_data[date_key]['total'] > 0:
                daily_data[date_key]['avg_load_percentage'] = round(
                    daily_data[date_key]['total_load_percentage'] / daily_data[date_key]['total'], 
                    2
                )
            del daily_data[date_key]['total_load_percentage']
            
        # Sort by date
        daily_result = list(daily_data.values())
        daily_result.sort(key=lambda x: x['date'])
        
        # Generate weekly data (group by week)
        weekly_data = {}
        for prediction in predictions:
            week_num = prediction.created_at.strftime('%U')
            year = prediction.created_at.strftime('%Y')
            week_key = f"{year}-W{week_num}"
            
            if week_key not in weekly_data:
                weekly_data[week_key] = {
                    'week': week_key,
                    'total': 0,
                    'overloaded': 0,
                    'safe': 0,
                    'avg_load_percentage': 0,
                    'total_load_percentage': 0
                }
                
            weekly_data[week_key]['total'] += 1
            weekly_data[week_key]['overloaded'] += 1 if prediction.is_overloaded else 0
            weekly_data[week_key]['safe'] += 0 if prediction.is_overloaded else 1
            weekly_data[week_key]['total_load_percentage'] += prediction.load_percentage
            
        # Calculate averages for weekly data
        for week_key in weekly_data:
            if weekly_data[week_key]['total'] > 0:
                weekly_data[week_key]['avg_load_percentage'] = round(
                    weekly_data[week_key]['total_load_percentage'] / weekly_data[week_key]['total'], 
                    2
                )
            del weekly_data[week_key]['total_load_percentage']
            
        # Sort by week
        weekly_result = list(weekly_data.values())
        weekly_result.sort(key=lambda x: x['week'])
        
        # Generate monthly data
        monthly_data = {}
        for prediction in predictions:
            month_key = prediction.created_at.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'month': month_key,
                    'total': 0,
                    'overloaded': 0,
                    'safe': 0,
                    'avg_load_percentage': 0,
                    'total_load_percentage': 0
                }
                
            monthly_data[month_key]['total'] += 1
            monthly_data[month_key]['overloaded'] += 1 if prediction.is_overloaded else 0
            monthly_data[month_key]['safe'] += 0 if prediction.is_overloaded else 1
            monthly_data[month_key]['total_load_percentage'] += prediction.load_percentage
            
        # Calculate averages for monthly data
        for month_key in monthly_data:
            if monthly_data[month_key]['total'] > 0:
                monthly_data[month_key]['avg_load_percentage'] = round(
                    monthly_data[month_key]['total_load_percentage'] / monthly_data[month_key]['total'], 
                    2
                )
            del monthly_data[month_key]['total_load_percentage']
            
        # Sort by month
        monthly_result = list(monthly_data.values())
        monthly_result.sort(key=lambda x: x['month'])
        
        # Return trends data
        return jsonify({
            'status': 'success',
            'data': {
                'daily': daily_result,
                'weekly': weekly_result,
                'monthly': monthly_result
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting prediction trends: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': 'Error retrieving prediction trends', 
            'error': str(e)
        }), 500 