import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Function to train a KNN model
def create_knn_model(data):
    # Separate the features (X) and target variable (y)
    X = data.drop(['id', 'diagnosis'], axis=1)
    y = data['diagnosis']

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize the features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors by default
    knn.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return knn, scaler, imputer

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

    # Train the KNN model and get the trained scaler and imputer
    knn_model, scaler, imputer = create_knn_model(data)

    # Ensure the directory 'model/' exists
    os.makedirs('model', exist_ok=True)

    # Save the trained KNN model to a file for later use
    with open('model/knn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)

    # Save the scaler to a file for consistent scaling during predictions
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save the imputer to handle missing values during predictions
    with open('model/imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

if __name__ == '__main__':
    main()
