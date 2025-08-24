from flask import request, render_template, jsonify, current_app
from routes import anomaly_bp
from models import db
from models.wine_anomaly import WineAnomaly
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Load pre-trained model + scaler
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'anomaly_model_0_80.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')


with open(model_path, "rb") as f:
    model = pickle.load(f)


with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


PLOT_FOLDER = os.path.join(base_dir, "static", "plots")
os.makedirs(PLOT_FOLDER, exist_ok=True)


@anomaly_bp.route('/')
def index():
    return render_template('index.html')

@anomaly_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    try:
        df = pd.read_csv(file)

        # Normalize column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.replace(' ', '_') 

        # Drop unwanted columns if any
        if 'quality' in df.columns:
            df.drop(columns=['quality'], inplace=True)

        expected_cols = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'color'
        ]

        X = df[expected_cols].copy()

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict anomalies and scores
        preds = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)

        # Convert predictions to human-readable string
        anomaly_results = ["Anomaly" if p == -1 else "Normal" for p in preds]

        results = []
        anomaly_count, normal_count = 0, 0

        for i, row in X.iterrows():
            anomaly_status = anomaly_results[i]

            if anomaly_status == "Anomaly":
                anomaly_count += 1
            else:
                normal_count += 1

            # Save anomaly record to DB
            anomaly = WineAnomaly(
                fixed_acidity=row['fixed_acidity'],
                volatile_acidity=row['volatile_acidity'],
                citric_acid=row['citric_acid'],
                residual_sugar=row['residual_sugar'],
                chlorides=row['chlorides'],
                free_sulfur_dioxide=row['free_sulfur_dioxide'],
                total_sulfur_dioxide=row['total_sulfur_dioxide'],
                density=row['density'],
                pH=row['pH'],  # match model column case
                sulphates=row['sulphates'],
                alcohol=row['alcohol'],
                color=row['color'],
                anomaly_score=float(scores[i]),
                anomaly_result=anomaly_status
            )
            db.session.add(anomaly)
            results.append(anomaly.to_dict())

        db.session.commit()

        # Plot histogram with counts on bars
        plt.figure(figsize=(6, 4))

        categories = ["Normal", "Anomaly", "Total"]
        counts = [normal_count, anomaly_count, normal_count + anomaly_count]
        colors = ["green", "red", "blue"]

        bars = plt.bar(categories, counts, color=colors)

        # Add text labels on top of bars
        for bar in bars:
           height = bar.get_height()
           plt.text(bar.get_x() + bar.get_width() / 2, height*1.01, str(int(height)), 
                    ha='center', va='bottom', fontsize=10)

        plt.title("Normal vs Anomaly Count")
        plt.ylabel("Count")
        plt.ylim(0, max(counts)*1.1)  # Add some headroom for text labels

        hist_path = os.path.join(PLOT_FOLDER, "histogram.png")
        plt.savefig(hist_path, bbox_inches="tight")
        plt.close()


        # Plot PCA if more than 2 features
        pca_path = None
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)
            df['pca_x'], df['pca_y'] = pcs[:, 0], pcs[:, 1]

            plt.figure(figsize=(5, 4))
            plt.scatter(df[df.index.isin([i for i, r in enumerate(anomaly_results) if r == 'Normal'])]['pca_x'],
                        df[df.index.isin([i for i, r in enumerate(anomaly_results) if r == 'Normal'])]['pca_y'],
                        c='green', label='Normal')
            plt.scatter(df[df.index.isin([i for i, r in enumerate(anomaly_results) if r == 'Anomaly'])]['pca_x'],
                        df[df.index.isin([i for i, r in enumerate(anomaly_results) if r == 'Anomaly'])]['pca_y'],
                        c='red', label='Anomaly')
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.title("PCA Visualization")
            pca_path = os.path.join(PLOT_FOLDER, "pca.png")
            plt.savefig(pca_path, bbox_inches="tight")
            plt.close()
         

        # Return JSON results with optional plot URLs for frontend display
        return jsonify(results)

    except pd.errors.EmptyDataError:
        return jsonify({"error": "Uploaded file is empty"}), 400
    except pd.errors.ParserError:
        return jsonify({"error": "Error parsing CSV file"}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to process file: {e}")
        return jsonify({"error": "An internal error occurred during processing"}), 500
