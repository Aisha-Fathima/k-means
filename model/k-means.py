import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

# Function to apply K-means clustering
def create_kmeans_model(data):
    # Separate the features (X) and drop unnecessary columns
    X = data.drop(['id', 'diagnosis'], axis=1)

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize the features for better clustering performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)  # Retain 10 principal components
    X_pca = pca.fit_transform(X_scaled)

    # Apply K-means clustering
    kmeans = KMeans(
        n_clusters=2,                 # Choose 2 clusters (e.g., benign vs malignant as a reference)
        random_state=42,              # Seed for reproducibility
        n_init=10                     # Number of initializations for the algorithm
    )
    kmeans.fit(X_pca)

    # Evaluate the clustering using silhouette score
    silhouette_avg = silhouette_score(X_pca, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg}")

    return kmeans, scaler, imputer, pca

# Function to load and preprocess the data
def get_clean_data():
    # Load the dataset
    data = pd.read_csv("data/data.csv")
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32'], axis=1, errors='ignore')
    # Map the target variable: Malignant ('M') as 1, Benign ('B') as 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    # Load the cleaned dataset
    data = get_clean_data()

    # Train the K-means model and get the trained scaler, imputer, and PCA
    kmeans_model, scaler, imputer, pca = create_kmeans_model(data)

    # Ensure the directory 'model/' exists
    os.makedirs('model', exist_ok=True)

    # Save the trained K-means model to a file for later use
    with open('model/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)

    # Save the scaler to a file for consistent scaling during predictions
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save the imputer to handle missing values during predictions
    with open('model/imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    # Save the PCA transformer for consistent dimensionality reduction
    with open('model/pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

if __name__ == '__main__':
    main()
