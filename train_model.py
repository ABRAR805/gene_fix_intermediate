import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import requests
import json
import time
import joblib

# Function to load mutation data from the provided JSON file
def load_mutation_data():
    # Replace this with loading from an actual data file or API response.
    # Example response structure (assumed based on provided JSON):
    mutation_data = {
        "mutations": [
            {
                "chr": "17",
                "pos": 7578189,
                "ref": "A",
                "alt": "C",
                "gene": "TP53",
                "consequence": "STOP_GAINED",
                "cadd_score": 15.181719,
                "cosmic_id": "COSM44505",
                "tumor_site": "stomach",
                "mutation_type": "SNV"
            },
            # More mutation data can be added here
        ]
    }
    return pd.DataFrame(mutation_data["mutations"])

# Function to fetch real-time data from myvariant.info API
def fetch_genomic_data(gene):
    api_url = f'https://myvariant.info/v1/query?q={gene}&fields=clinvar'
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Ensure we raise an exception for a bad status code
        data = response.json()
        
        # Check if there is valid data in the response
        if "hits" in data and len(data["hits"]) > 0:
            clinvar_data = data["hits"][0].get("clinvar", {})
            return clinvar_data
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for gene {gene}: {e}")
        return None

# Feature engineering function
def feature_engineering(mutation_df):
    # Extract relevant features from the mutation data
    mutation_df["is_stop_gained"] = mutation_df["consequence"].apply(lambda x: 1 if x == "STOP_GAINED" else 0)
    mutation_df["cadd_score"] = mutation_df["cadd_score"].fillna(mutation_df["cadd_score"].mean())
    
    # Tumor site encoded as categorical
    mutation_df["tumor_site"] = mutation_df["tumor_site"].astype('category').cat.codes
    
    # Fetch ClinVar data for each mutation's gene and add relevant features
    mutation_df["clinvar_significance"] = mutation_df["gene"].apply(lambda gene: fetch_clinvar_significance(gene))
    
    # Return the modified dataframe
    return mutation_df

# Function to extract ClinVar significance from the API response
def fetch_clinvar_significance(gene):
    clinvar_data = fetch_genomic_data(gene)
    if clinvar_data and "significance" in clinvar_data:
        # Process or map significance to numeric values for easier modeling
        if "Pathogenic" in clinvar_data["significance"]:
            return 1  # Pathogenic
        elif "Benign" in clinvar_data["significance"]:
            return 0  # Benign
        else:
            return -1  # Unknown significance
    else:
        return -1  # Unknown if no ClinVar data is available

# Train a machine learning model
def train_model(mutation_df):
    # Define features (X) and target (y)
    X = mutation_df[["cadd_score", "is_stop_gained", "tumor_site", "clinvar_significance"]]  # Features you want to use
    
    # Use LabelEncoder for categorical target mutation_type
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(mutation_df["mutation_type"])  # Encoding the target mutation_type

    # Ensure the dataset has more than 1 sample for splitting
    if len(mutation_df) > 1:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        # If only one sample, use it for both training and testing
        X_train, X_test, y_train, y_test = X, X, y, y
        print("Not enough data for train/test split. Using the same data for both training and testing.")

    # Train a Random Forest Classifier model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model and label encoder for future use
    joblib.dump(clf, 'mutation_model.pkl')  # Save the classifier model
    joblib.dump(label_encoder, 'labelencoder.pkl')  # Save the label encoder

    return clf, label_encoder

# Main function
if __name__ == "__main__":
    # Step 1: Load mutation data (replace this with actual data)
    mutation_df = load_mutation_data()
    
    # Step 2: Feature Engineering (adding more useful features to the dataset)
    mutation_df = feature_engineering(mutation_df)
    
    # Step 3: Train the model with the mutation dataset
    model, label_encoder = train_model(mutation_df)

    print("Model training complete and saved to mutation_model.pkl and labelencoder.pkl")
