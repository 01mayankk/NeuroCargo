from flask import jsonify

def register_error_handlers(app):
    """Register error handlers with the Flask app."""
    
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({
            'status': 'error',
            'message': 'Bad request',
            'error': str(e)
        }), 400

    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({
            'status': 'error',
            'message': 'Unauthorized',
            'error': str(e)
        }), 401

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({
            'status': 'error',
            'message': 'Forbidden',
            'error': str(e)
        }), 403

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            'status': 'error',
            'message': 'Resource not found',
            'error': str(e)
        }), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        app.logger.error(f"Internal server error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e)
        }), 500

    # JWT specific errors
    @app.errorhandler(422)
    def handle_unprocessable_entity(e):
        return jsonify({
            'status': 'error',
            'message': 'Unprocessable entity',
            'error': str(e)
        }), 422 