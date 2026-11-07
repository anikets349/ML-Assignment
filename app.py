import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_DIR = "models"
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Gaussian NB",
    "Multinomial NB",
    "SVM"
]
PRICE_RANGES = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

def engineer_features(df):
    """Applies the exact same feature engineering as in the training notebook."""
    df = df.copy()
    # Ensure columns exist before engineering to avoid errors on single row inputs
    if 'fc' in df.columns:
        df["fc_log"] = np.log1p(df["fc"])
    if 'px_height' in df.columns and 'px_width' in df.columns:
        df["px_area"] = df["px_height"] * df["px_width"]
    if 'sc_h' in df.columns and 'sc_w' in df.columns:
        df["screen_area"] = df["sc_h"] * df["sc_w"]
    if 'fc_log' in df.columns and 'pc' in df.columns:
        df["cam_px"] = df["fc_log"] + df["pc"]
    
    # Replace infinities with NaN
    return df.replace([np.inf, -np.inf], np.nan)

@st.cache_resource
def load_model(model_name):
    """Loads the trained model pipeline from disk."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

st.set_page_config(page_title="Mobile Price Predictor", layout="wide")
st.title("📱 Mobile Price Range Predictor")
st.markdown("Enter the specifications of the mobile phone to predict its price range.")

# Sidebar for File Uploads
st.sidebar.header("📂 Dataset Upload")
train_file = st.sidebar.file_uploader("Upload Train CSV", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

# Sidebar for model selection
st.sidebar.header("⚙️ Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", MODEL_NAMES)

# Input form (Sliders and bars removed)
st.header("Mobile Specifications")
col1, col2, col3, col4 = st.columns(4)

with col1:
    battery_power = st.number_input("Battery Power (mAh)", value=1200)
    blue = st.selectbox("Bluetooth", ["Yes", "No"])
    clock_speed = st.number_input("Clock Speed (GHz)", value=1.5)
    dual_sim = st.selectbox("Dual SIM", ["Yes", "No"])

with col2:
    fc = st.number_input("Front Camera (MP)", value=2)
    four_g = st.selectbox("4G", ["Yes", "No"])
    int_memory = st.number_input("Internal Memory (GB)", value=32)
    m_dep = st.number_input("Mobile Depth (cm)", value=0.5)

with col3:
    # Changed to number input as mobile weight is typically numeric in this dataset
    mobile_wt = st.number_input("Mobile Weight (g)", value=150) 
    n_cores = st.number_input("Number of Cores", value=4)
    pc = st.number_input("Primary Camera (MP)", value=10)
    px_height = st.number_input("Pixel Resolution Height", value=800)

with col4:
    px_width = st.number_input("Pixel Resolution Width", value=1200)
    ram = st.number_input("RAM (MB)", value=2000)
    sc_h = st.number_input("Screen Height (cm)", value=12)
    sc_w = st.number_input("Screen Width (cm)", value=7)

col5, col6 = st.columns(2)
with col5:
    talk_time = st.number_input("Talk Time (hours)", value=10)
    three_g = st.selectbox("3G", ["Yes", "No"])

with col6:
    touch_screen = st.selectbox("Touch Screen", ["Yes", "No"])
    wifi = st.selectbox("WiFi", ["Yes", "No"])

# Map UI inputs to the exact format expected by the model
input_data = {
    "battery_power": battery_power,
    "blue": 1 if blue == "Yes" else 0,
    "clock_speed": clock_speed,
    "dual_sim": 1 if dual_sim == "Yes" else 0,
    "fc": fc,
    "four_g": 1 if four_g == "Yes" else 0,
    "int_memory": int_memory,
    "m_dep": m_dep,
    "mobile_wt": mobile_wt,
    "n_cores": n_cores,
    "pc": pc,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": sc_h,
    "sc_w": sc_w,
    "talk_time": talk_time,
    "three_g": 1 if three_g == "Yes" else 0,
    "touch_screen": 1 if touch_screen == "Yes" else 0,
    "wifi": 1 if wifi == "Yes" else 0
}

# Create DataFrame and apply feature engineering
input_df = pd.DataFrame([input_data])
input_df_eng = engineer_features(input_df)

# Single Prediction button
if st.button("🔮 Predict Single Device"):
    try:
        model = load_model(selected_model)
        prediction = model.predict(input_df_eng)
        probabilities = model.predict_proba(input_df_eng)
        predicted_class = int(prediction[0])
        
        st.subheader("Prediction Result")
        st.success(f"**Predicted Price Range:** {predicted_class} ({PRICE_RANGES[predicted_class]})")
        
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
            probabilities,
            columns=[f"Class {i} ({PRICE_RANGES[i]})" for i in range(4)],
            index=["Prediction"]
        )
        st.dataframe(prob_df.style.format("{:.2%}"))
        st.bar_chart(prob_df.T)
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.markdown("---")

if test_file is not None:
    st.header("📊 Test Set Results")
    try:
        df_test = pd.read_csv(test_file)
        
        # Separate features and target if target exists
        has_labels = 'price_range' in df_test.columns
        if has_labels:
            X_test = df_test.drop('price_range', axis=1)
            y_test = df_test['price_range']
        else:
            X_test = df_test.copy()
            
        # Feature engineering
        X_test_eng = engineer_features(X_test)
        
        # Load model and predict
        model = load_model(selected_model)
        predictions = model.predict(X_test_eng)
        
        # Display Predictions
        st.subheader("Predicted Price Ranges")
        result_df = X_test.copy() # Show original features
        result_df['Predicted_Price_Range'] = predictions
        result_df['Predicted_Category'] = result_df['Predicted_Price_Range'].map(PRICE_RANGES)
        
        # Show first 10 rows to avoid clutter
        st.dataframe(result_df.head(10))
        st.write(f"*Showing first 10 of {len(result_df)} predictions.*")
        
        # Evaluation Metrics
        st.subheader("📈 Model Evaluation Metrics")
        
        y_true = None
        eval_source = ""
        
        # Use Test labels if available, otherwise fallback to Train labels
        if has_labels:
            y_true = y_test
            eval_source = "Test Set"
        elif train_file is not None:
            st.info("⚠️ Test CSV does not contain 'price_range' column. Calculating metrics using the **Train CSV** instead.")
            df_train = pd.read_csv(train_file)
            X_train = df_train.drop('price_range', axis=1)
            y_true = df_train['price_range']
            X_train_eng = engineer_features(X_train)
            # Re-predict on train set for metrics
            predictions_eval = model.predict(X_train_eng)
            eval_source = "Train Set (Fallback)"
        else:
            st.warning("Cannot display evaluation metrics. The Test CSV has no labels, and no Train CSV was uploaded.")
            
        if y_true is not None:
            # If we used the fallback, we need the predictions for the train set
            if eval_source == "Train Set (Fallback)":
                y_pred = predictions_eval
            else:
                y_pred = predictions
                
            # 1. Accuracy
            acc = accuracy_score(y_true, y_pred)
            st.metric("Overall Accuracy", f"{acc:.2%}")
            
            col_rep, col_cm = st.columns(2)
            
            with col_rep:
                st.subheader("Classification Report")
                target_names = [PRICE_RANGES[i] for i in range(4)]
                report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df)
                
            with col_cm:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=target_names, 
                            yticklabels=target_names,
                            cbar=False)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title(f'Confusion Matrix ({eval_source})')
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Error processing Test CSV: {e}")
else:
    st.info("👉 Upload the **Test CSV** in the sidebar to view batch predictions and evaluation metrics.")