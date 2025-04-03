from datetime import datetime
from .. import db

class Vehicle(db.Model):
    """Vehicle model for storing vehicle data."""
    __tablename__ = 'vehicles'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    vehicle_type = db.Column(db.String(50), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    max_load_capacity = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='vehicle', lazy='dynamic', cascade='all, delete-orphan')
    
    def __init__(self, name, vehicle_type, weight, max_load_capacity, user_id, description=None):
        self.name = name
        self.vehicle_type = vehicle_type
        self.weight = weight
        self.max_load_capacity = max_load_capacity
        self.user_id = user_id
        self.description = description
    
    def to_dict(self):
        """Convert vehicle model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'vehicle_type': self.vehicle_type,
            'weight': self.weight,
            'max_load_capacity': self.max_load_capacity,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'user_id': self.user_id
        }
    
    def __repr__(self):
        return f'<Vehicle {self.name} ({self.vehicle_type})>'


class Prediction(db.Model):
    """Prediction model for storing prediction data."""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'), nullable=False)
    passenger_count = db.Column(db.Integer, nullable=False)
    cargo_weight = db.Column(db.Float, nullable=False)
    region = db.Column(db.String(50), nullable=True)
    road_condition = db.Column(db.String(50), nullable=True)
    weather = db.Column(db.String(50), nullable=True)
    is_overloaded = db.Column(db.Boolean, nullable=False)
    load_percentage = db.Column(db.Float, nullable=False)
    risk_assessment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    remaining_capacity = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, user_id, vehicle_id, passenger_count, cargo_weight, is_overloaded, 
                 load_percentage, risk_assessment, confidence, remaining_capacity, 
                 region=None, road_condition=None, weather=None):
        self.user_id = user_id
        self.vehicle_id = vehicle_id
        self.passenger_count = passenger_count
        self.cargo_weight = cargo_weight
        self.region = region
        self.road_condition = road_condition
        self.weather = weather
        self.is_overloaded = is_overloaded
        self.load_percentage = load_percentage
        self.risk_assessment = risk_assessment
        self.confidence = confidence
        self.remaining_capacity = remaining_capacity
    
    def to_dict(self):
        """Convert prediction model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'vehicle_id': self.vehicle_id,
            'passenger_count': self.passenger_count,
            'cargo_weight': self.cargo_weight,
            'region': self.region,
            'road_condition': self.road_condition,
            'weather': self.weather,
            'is_overloaded': self.is_overloaded,
            'load_percentage': self.load_percentage,
            'risk_assessment': self.risk_assessment,
            'confidence': self.confidence,
            'remaining_capacity': self.remaining_capacity,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Prediction {self.id} for Vehicle {self.vehicle_id}>' 