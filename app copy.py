from flask import Flask
from models import db
from routes import anomaly_bp
import os

def create_app():
    app = Flask(__name__)

    app.config.from_object('config.Config')

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    app.register_blueprint(anomaly_bp)

    # Create SQLite DB and tables if they don't exist
    with app.app_context():
        db.create_all()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
