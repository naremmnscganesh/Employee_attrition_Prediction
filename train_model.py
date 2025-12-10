import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_preprocess_data(filepath):
    """Loads data and performs preprocessing for selected features."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Select only the top 5 important features + Target
    selected_features = ['Age', 'MonthlyIncome', 'OverTime', 'TotalWorkingYears', 'YearsAtCompany', 'Attrition']
    print(f"Selecting features: {selected_features}")
    df = df[selected_features]

    # 2. Encode Target Variable 'Attrition'
    print("Encoding target variable 'Attrition'...")
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'])
    
    # 3. Categorical Encoding (One-Hot Encoding)
    # Identify categorical columns (only OverTime in this subset)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {cat_cols}")
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    print(f"Data shape after preprocessing: {df.shape}")
    return df

def train_and_evaluate(df):
    """Trains a Random Forest model and evaluates it."""
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Split data
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    
    # Evaluation
    print("\nModel Evaluation:")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("\nFeature Importances:")
    print(importances.sort_values(ascending=False))

    # Save model and columns
    print("\nSaving model and columns...")
    joblib.dump(rf, 'model.pkl')
    joblib.dump(list(X.columns), 'model_columns.pkl')
    print("Model saved to model.pkl")
    print("Columns saved to model_columns.pkl")

if __name__ == "__main__":
    filepath = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    try:
        df = load_and_preprocess_data(filepath)
        train_and_evaluate(df)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")
