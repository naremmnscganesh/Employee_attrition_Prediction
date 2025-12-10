# Employee Attrition Prediction

A Machine Learning powered web application to predict employee attrition. This tool helps HR and management identify employees who might be at risk of leaving the company based on key factors.

## 🚀 Features

*   **Machine Learning Model**: Built using Random Forest Classifier.
*   **Simplified Inputs**: Uses the top 5 most impactful features for prediction:
    1.  **Monthly Income**
    2.  **Age**
    3.  **Over Time**
    4.  **Total Working Years**
    5.  **Years At Company**
*   **Web Interface**: User-friendly Flask application to get real-time predictions.
*   **Interactive UI**: Clean, modern design with immediate feedback.

## 📋 Prerequisites

Ensure you have Python installed. You will need the following libraries:

```bash
pip install pandas numpy scikit-learn flask
```

## 🛠️ Installation & Usage

1.  **Clone or Download** the repository.
2.  **Navigate** to the project directory:
    ```bash
    cd "e:/Employee attrition prediction"
    ```

### 1. Training the Model (Optional)
The project comes with a pre-trained model (`model.pkl`). If you want to retrain it:

```bash
python train_model.py
```
This will:
-   Load the dataset (`data/WA_Fn-UseC_-HR-Employee-Attrition.csv`).
-   Preprocess the data and select top 5 features.
-   Train the Random Forest model.
-   Save `model.pkl` and `model_columns.pkl`.

### 2. Running the Web Application
Start the Flask server:

```bash
python app.py
```

Open your browser and go to:
**http://127.0.0.1:5000**

## 📂 Project Structure

```
Employee attrition prediction/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
├── static/
│   └── style.css                              # CSS Styling
├── templates/
│   └── index.html                             # HTML Frontend
├── app.py                                     # Flask Backend
├── train_model.py                             # ML Training Script
├── model.pkl                                  # Saved Model
├── model_columns.pkl                          # Saved Feature Columns
└── README.md                                  # Documentation
```

## 📊 Model Performance
-   **Algorithm**: Random Forest
-   **Accuracy**: ~81% (on simplified 5-feature subset)
-   **Key Insight**: OverTime and Monthly Income are among the strongest predictors of attrition.
