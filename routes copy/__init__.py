from flask import Blueprint

anomaly_bp = Blueprint('anomaly', __name__)

from routes import anomaly_routes
