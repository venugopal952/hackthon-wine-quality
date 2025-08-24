from WineQualityAnomalyDetector import WineQualityAnomalyDetector

# 2. Provide paths to your red and white wine CSV files
red_wine_path = '/Users/maldann.gattannagari/Downloads/hackthon-final/Wine-quality-dashboad/ML-model-training/dataset/winequality-red.csv'
white_wine_path = '/Users/maldann.gattannagari/Downloads/hackthon-final/Wine-quality-dashboad/ML-model-training/dataset/winequality-white.csv'

# 3. Create an instance of the detector
detector = WineQualityAnomalyDetector(red_wine_path, white_wine_path)

# 4. Load and prepare data
detector.load_and_prepare_data()

# 5. Scale features and get feature names
scaled_data, feature_names = detector.scale_features()

# 6. Extract labels for training
labels = detector.data['defect']

# 9. Train the Isolation Forest model (this may take some time for many iterations)
detector.train_isolation_forest(scaled_data, labels, iterations=50000)


# 7. Visualize correlation heatmap and feature histograms
detector.plot_feature_correlation(scaled_data, feature_names)
detector.plot_feature_histograms(detector.data.drop(columns=['defect']))

# 8. Plot anomaly score distribution and PCA visualization
detector.plot_anomaly_scores(scaled_data)
detector.plot_pca(scaled_data)

# 10. Test on a new wine sample (replace with actual feature values)
new_wine_sample = {
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.5,
    "citric_acid": 0.2,
    "residual_sugar": 8.0,
    "chlorides": 0.07,
    "free_sulfur_dioxide": 15,
    "total_sulfur_dioxide": 40,
    "density": 0.996,
    "pH": 3.3,
    "sulphates": 0.5,
    "alcohol": 10.5,
    "color": 1
}

prediction, score = detector.test_new_entry(new_wine_sample)
print(f"Prediction: {'Anomaly' if prediction == -1 else 'Normal'}, Score: {score}")
