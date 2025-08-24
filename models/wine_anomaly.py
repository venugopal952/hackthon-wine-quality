from models import db

class WineAnomaly(db.Model):
    __tablename__ = 'wine_anomalies'
    id = db.Column(db.Integer, primary_key=True)
    fixed_acidity = db.Column(db.Float)
    volatile_acidity = db.Column(db.Float)
    citric_acid = db.Column(db.Float)
    residual_sugar = db.Column(db.Float)
    chlorides = db.Column(db.Float)
    free_sulfur_dioxide = db.Column(db.Float)
    total_sulfur_dioxide = db.Column(db.Float)
    density = db.Column(db.Float)
    pH = db.Column(db.Float)
    sulphates = db.Column(db.Float)
    alcohol = db.Column(db.Float)
    color = db.Column(db.Float)
    anomaly_score = db.Column(db.Float)
    anomaly_result = db.Column(db.String(20))

    def to_dict(self):
        return {
            'id': self.id,
            'fixed_acidity': self.fixed_acidity,
            'volatile_acidity': self.volatile_acidity,
            'citric_acid': self.citric_acid,
            'residual_sugar': self.residual_sugar,
            'chlorides': self.chlorides,
            'free_sulfur_dioxide': self.free_sulfur_dioxide,
            'total_sulfur_dioxide': self.total_sulfur_dioxide,
            'density': self.density,
            'pH': self.pH,
            'sulphates': self.sulphates,
            'alcohol': self.alcohol,
            'color': self.color,
            'anomaly_score': self.anomaly_score,
            'anomaly_result': self.anomaly_result,
        }
