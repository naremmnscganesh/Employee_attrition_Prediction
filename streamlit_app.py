import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🏢")

def load_persistence_artifacts():
    """Loads the model and feature columns."""
    try:
        model = joblib.load("model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'model.pkl' and 'model_columns.pkl' exist.")
        return None, None

def main():
    st.title("🏢 Employee Attrition Predictor")
    st.markdown("""
    Enter the employee's details below to predict the likelihood of attrition.
    This model uses the **top 5 key factors** influencing employee turnover.
    """)

    model, model_columns = load_persistence_artifacts()

    if model is None:
        return

    # --- Input Form ---
    with st.form("prediction_form"):
        st.subheader("Employee Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
            over_time = st.selectbox("Over Time", ["No", "Yes"])
        
        with col2:
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=60, value=5)
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=60, value=3)

        submit = st.form_submit_button("Predict Attrition")

    # --- Prediction Logic ---
    if submit:
        # Create a dict of inputs
        input_data = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'OverTime': over_time,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company
        }

        # create dataframe
        input_df = pd.DataFrame([input_data])
        
        # Preprocessing
        # 1. Get dummies (handles OverTime -> OverTime_Yes)
        input_df = pd.get_dummies(input_df)
        
        # 2. Align with model columns (missing cols become 0)
        # Reindex ensures we have exact same columns as training
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        # Display Result
        st.markdown("---")
        st.subheader("Prediction Result")
        
        result_class = prediction[0] # 1 = Yes, 0 = No
        confidence = probability[0][1] if result_class == 1 else probability[0][0]
        
        if result_class == 1:
            st.error(f"⚠️ **Yes**, this employee is likely to leave.")
            st.metric(label="Confidence Level", value=f"{confidence:.2%}")
        else:
            st.success(f"✅ **No**, this employee is likely to stay.")
            st.metric(label="Confidence Level", value=f"{confidence:.2%}")

if __name__ == "__main__":
    main()
