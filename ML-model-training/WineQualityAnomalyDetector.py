import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA

class WineQualityAnomalyDetector:
    def __init__(self, red_path, white_path):
        """
        Initialize with dataset file paths.
        """
        self.red_path = red_path
        self.white_path = white_path
        self.data = None
        self.scaler = None
        self.model = None

    def load_and_prepare_data(self):
        print("start")
        """
        Loads red and white wine datasets, merges them,
        processes features and creates synthetic defect labels.
        """
        red = pd.read_csv(self.red_path, sep=';', quotechar='"')
        white = pd.read_csv(self.white_path, sep=';', quotechar='"')

        # Drop 'quality' column if present
        for df in [red, white]:
            if 'quality' in df.columns:
                df.drop(columns=['quality'], inplace=True)

        # Add color feature 1 for red, 2 for white
        red['color'] = 1
        white['color'] = 2

        # Combine datasets
        self.data = pd.concat([red, white], ignore_index=True)

        # Clean column names
        self.data.columns = self.data.columns.str.strip().str.replace(' ', '_')

        self.data['defect'] = self.data.apply(self.synthetic_defect_label, axis=1)

    @staticmethod
    def synthetic_defect_label(row):
        """
        Create a synthetic defect label based on feature thresholds.
        """
        if (row['residual_sugar'] > 10) or (row['sulphates'] < 0.4) or (row['alcohol'] < 9):
            return 1
        else:
            return 0

    def scale_features(self):
        """
        Scale features using StandardScaler.
        Saves the scaler to 'scaler.pkl'.
        """
        features = self.data.drop(columns=['defect'])
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(features)

        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        return scaled, features.columns

    def train_isolation_forest(self, scaled_data, labels, iterations=10000):
        """
        Performs hyperparameter tuning for Isolation Forest.
        Saves best model to 'best_isolation_forest_model.pkl'.
        """
        best_f1 = -np.inf
        best_f1_recall = -np.inf
        best_f1_precision = -np.inf
        best_f1_accuracy = -np.inf

        best_model = None
        best_params = None

        contamination_values = np.linspace(0.1, 0.4, 100)
        n_estimators_choices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,30,40,50,60,70,80,90,100,150,200,300,400,]
        max_samples_choices = [0.1,0.2,0.3,0.5, 0.6, 0.7, 0.8]
        max_features_choices = [0.3,0.4, 0.5, 0.6, 0.7, 0.8,0.9,1]

        """
        Best Model Parameters Found:
{'n_estimators': np.int64(1), 'max_samples': np.float64(0.3), 'contamination': np.float64(0.3848484848484849), 'max_features': np.float64(1.0), 'random_state': 35}
Best F1 Score: 0.7162497182781159
Best F1 Recall: 0.7650457390467019
Best F1 Precision: 0.6733050847457627
Best F1 Accuracy: 0.8062182545790365

"""
        for iteration in range(iterations):
            params = {
                'n_estimators': np.random.choice(n_estimators_choices),
                'max_samples': np.random.choice(max_samples_choices),
                'contamination': np.random.choice(contamination_values),
                'max_features': np.random.choice(max_features_choices),
                'random_state': np.random.randint(22, 65)
            }
            model = IsolationForest(**params)
            preds = model.fit_predict(scaled_data)
            preds_binary = (preds == -1).astype(int)

            precision = precision_score(labels, preds_binary)
            recall = recall_score(labels, preds_binary)
            accuracy = accuracy_score(labels, preds_binary)
            f1 = f1_score(labels, preds_binary, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_f1_recall = recall
                best_f1_precision = precision
                best_f1_accuracy = accuracy
                best_model = model
                best_params = params
                with open('best_isolation_forest_model.pkl', 'wb') as f:
                    pickle.dump(best_model, f)

            if (iteration + 1) % 500 == 0:
                print(f"Iteration {iteration + 1}/{iterations} | Current Best F1: {best_f1:.4f}  | Current best_f1_recall: {best_f1_recall:.4f}  | Current best_f1_precision: {best_f1_precision:.4f}  | Current best_f1_accuracy: {best_f1_accuracy:.4f}")

        print("\nBest Model Parameters Found:")
        print(best_params)
        print("Best F1 Score:", best_f1)
        print("Best F1 Recall:", best_f1_recall)
        print("Best F1 Precision:", best_f1_precision)
        print("Best F1 Accuracy:", best_f1_accuracy)
        self.model = best_model

    def plot_feature_correlation(self, scaled_data, feature_names):
        """
        Plot and show feature correlation heatmap.
        """
        plt.figure(figsize=(12, 10))
        corr_df = pd.DataFrame(scaled_data, columns=feature_names).corr()
        sns.heatmap(corr_df, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def plot_feature_histograms(self, features):
        """
        Plot histograms for each feature.
        """
        features.hist(bins=30, figsize=(15, 12))
        plt.suptitle("Feature Distributions")
        plt.show()

    def plot_anomaly_scores(self, scaled_data):
        """
        Plot histogram of anomaly scores from the best model.
        """
        anomaly_scores = -self.model.decision_function(scaled_data)
        plt.figure(figsize=(8,6))
        sns.histplot(anomaly_scores, bins=50, kde=True)
        plt.title('Anomaly Scores Distribution (Higher = More Anomalous)')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.show()

    def plot_pca(self, scaled_data):
        """
        Generate and plot PCA visualization of anomalies.
        """
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled_data)
        plt.figure(figsize=(10,8))
        scatter = plt.scatter(pcs[:, 0], pcs[:, 1],
                              c=(self.model.predict(scaled_data) == -1),
                              cmap='coolwarm', alpha=0.6)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Anomalies in 2D PCA Projection (Red = Anomaly)')
        plt.colorbar(scatter, label='Anomaly (1) / Normal (0)')
        plt.show()

    def test_new_entry(self, new_entry):
        """
        Predict anomaly for a new single data entry.
        new_entry: dict of feature values.
        """
        new_df = pd.DataFrame([new_entry])
        new_scaled = self.scaler.transform(new_df)
        prediction = self.model.predict(new_scaled)
        anomaly_score = self.model.decision_function(new_scaled)[0]

        print("\nNew Entry Prediction:", "Anomaly" if prediction[0] == -1 else "Normal")
        print("New Entry Anomaly Score:", anomaly_score)
        return prediction[0], anomaly_score