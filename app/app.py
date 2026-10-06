# ============================================
# BREAST CANCER CLASSIFICATION WEB APPLICATION
# ============================================

# Import Streamlit for building the web interface.
import streamlit as st

# Import pandas for creating the input data table
# expected by the trained machine learning pipeline.
import pandas as pd

# Import joblib for loading the saved Random Forest pipeline.
import joblib

# Import Path for safe file and folder handling.
from pathlib import Path


# ============================================
# PAGE CONFIGURATION
# ============================================

# Configure the browser tab and application layout.
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="centered"
)


# ============================================
# PROJECT AND MODEL PATHS
# ============================================

# app.py is inside the "app" folder.
# Therefore, move one level up to reach the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build the path to the saved tuned Random Forest pipeline.
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_pipeline.pkl"
)


# ============================================
# LOAD TRAINED MODEL
# ============================================

# Cache the model so Streamlit does not reload the
# .pkl file every time the user changes a slider.
@st.cache_resource
def load_model():

    # Load and return the trained Random Forest pipeline.
    return joblib.load(MODEL_PATH)


# Load the model when the application starts.
model = load_model()

# ============================================
# SIDEBAR - MODEL INFORMATION
# ============================================

# Display information about the deployed ML model
# in a separate sidebar.
with st.sidebar:

    # Sidebar title.
    st.header("Model Information")

    # Explain which model is being used.
    st.write(
        "This prototype uses the **Tuned Random Forest** "
        "selected from the group's final model comparison."
    )

    st.divider()

    # Display the final held-out test performance
    # obtained during model evaluation.
    st.subheader("Final Test Performance")

    st.metric("Accuracy", "97.84%")
    st.metric("Precision", "95.92%")
    st.metric("Recall", "97.92%")
    st.metric("F1-Score", "96.91%")
    st.metric("ROC-AUC", "99.34%")

    st.divider()

    # Display important information about the model.
    st.subheader("Model Setup")

    st.write("**Algorithm:** Random Forest")
    st.write("**Trees:** 200")
    st.write("**Maximum depth:** 5")
    st.write("**Class weight:** Balanced")
    st.write("**False negatives on test set:** 1")

    st.divider()

    # Explain the dataset used to build the prototype.
    st.subheader("Dataset")

    st.write(
        "**Breast Cancer Wisconsin (Original)**"
    )

    st.write(
        "9 predictive cell characteristics"
    )

    st.write(
        "691 observations used after removing "
        "8 exact duplicate observations."
    )

    st.divider()

    # Add an ethical-use reminder.
    st.subheader("Important")

    st.caption(
        "This prototype demonstrates an academic "
        "machine-learning workflow. It has not been "
        "clinically validated and must not be used "
        "for diagnosis or treatment decisions."
    )

# ============================================
# APPLICATION HEADER
# ============================================

# Display the main application title.
st.title("🔬 Breast Cancer Classification")

# Explain the purpose of the application.
st.write(
    "An AI/ML demonstration for classifying breast cancer "
    "samples as **Benign** or **Malignant**."
)

# Show which model powers the application.
st.caption(
    "Prediction model: Tuned Random Forest"
)

st.divider()


# ============================================
# SAMPLE FEATURE INPUT SECTION
# ============================================

st.subheader("Sample Features")

st.write(
    "Enter the nine cell characteristics below. "
    "Each feature uses the 1–10 scale from the "
    "Breast Cancer Wisconsin (Original) dataset."
)


# ============================================
# FEATURE INPUTS
# ============================================

# Divide the feature inputs into two columns
# to create a cleaner and more compact interface.
column_1, column_2 = st.columns(2)


# ============================================
# LEFT COLUMN
# ============================================

with column_1:

    # Collect Clump Thickness.
    clump_thickness = st.slider(
        "Clump Thickness",
        min_value=1,
        max_value=10,
        value=5
    )

    # Collect Uniformity of Cell Size.
    uniformity_cell_size = st.slider(
        "Uniformity of Cell Size",
        min_value=1,
        max_value=10,
        value=5
    )

    # Collect Uniformity of Cell Shape.
    uniformity_cell_shape = st.slider(
        "Uniformity of Cell Shape",
        min_value=1,
        max_value=10,
        value=5
    )

    # Collect Marginal Adhesion.
    marginal_adhesion = st.slider(
        "Marginal Adhesion",
        min_value=1,
        max_value=10,
        value=5
    )

    # Collect Single Epithelial Cell Size.
    single_epithelial_cell_size = st.slider(
        "Single Epithelial Cell Size",
        min_value=1,
        max_value=10,
        value=5
    )


# ============================================
# RIGHT COLUMN
# ============================================

with column_2:

    # Collect Bare Nuclei.
    bare_nuclei = st.slider(
        "Bare Nuclei",
        min_value=1,
        max_value=10,
        value=5
    )

    # Collect Bland Chromatin.
    bland_chromatin = st.slider(
        "Bland Chromatin",
        min_value=1,
        max_value=10,
        value=5
    )

    # Collect Normal Nucleoli.
    normal_nucleoli = st.slider(
        "Normal Nucleoli",
        min_value=1,
        max_value=10,
        value=5
    )

    # Collect Mitoses.
    mitoses = st.slider(
        "Mitoses",
        min_value=1,
        max_value=10,
        value=5
    )

# ============================================
# CREATE PREDICTION BUTTON
# ============================================

predict_button = st.button(
    "Predict Classification",
    type="primary",
    use_container_width=True
)


# ============================================
# MAKE PREDICTION
# ============================================

if predict_button:

    # Create one observation using the exact feature
    # names and order used during model training.
    input_data = pd.DataFrame(
        [[
            clump_thickness,
            uniformity_cell_size,
            uniformity_cell_shape,
            marginal_adhesion,
            single_epithelial_cell_size,
            bare_nuclei,
            bland_chromatin,
            normal_nucleoli,
            mitoses
        ]],
        columns=[
            "Clump_Thickness",
            "Uniformity_Cell_Size",
            "Uniformity_Cell_Shape",
            "Marginal_Adhesion",
            "Single_Epithelial_Cell_Size",
            "Bare_Nuclei",
            "Bland_Chromatin",
            "Normal_Nucleoli",
            "Mitoses"
        ]
    )

    # Use the saved tuned Random Forest pipeline
    # to predict the class.
    prediction = model.predict(input_data)[0]

    # Obtain predicted class probabilities.
    probabilities = model.predict_proba(input_data)[0]

    # Probability for Class 0 (Benign).
    benign_probability = probabilities[0]

    # Probability for Class 1 (Malignant).
    malignant_probability = probabilities[1]


    # ============================================
    # DISPLAY PREDICTION RESULT
    # ============================================


    # ============================================
    # DISPLAY ENTERED SAMPLE
    # ============================================

    # Allow the user to inspect the exact feature values
    # that were sent to the machine-learning model.
    with st.expander("View Entered Sample"):

        # Create a copy of the model input for display.
        display_data = input_data.copy()

        # Replace programming-style column names with
        # more readable names for the user interface.
        display_data.columns = [
            "Clump Thickness",
            "Uniformity Cell Size",
            "Uniformity Cell Shape",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses"
        ]

        # Display the exact values entered by the user.
        st.dataframe(
            display_data,
            hide_index=True,
            use_container_width=True
        )


       # ============================================
    # DISPLAY PREDICTION RESULT
    # ============================================

    st.divider()
    st.subheader("Prediction Result")

    # Class 0 represents Benign.
    if prediction == 0:
        st.success("Prediction: BENIGN")

    # Class 1 represents Malignant.
    else:
        st.error("Prediction: MALIGNANT")


    # ============================================
    # DISPLAY MODEL PROBABILITIES
    # ============================================

    st.write("### Model Output")

    # Display the estimated probability for each class.
    st.write(
        f"Benign probability: **{benign_probability * 100:.2f}%**"
    )

    st.write(
        f"Malignant probability: **{malignant_probability * 100:.2f}%**"
    )

    # Display a simple probability bar.
    st.progress(float(malignant_probability))

    st.caption(
        "The probabilities shown are model outputs and "
        "should not be interpreted as clinical risk estimates."
    )


# ============================================
# DISCLAIMER
# ============================================

st.divider()

st.warning(
    "Educational AI/ML demonstration only. "
    "This application is not intended for clinical diagnosis "
    "or medical decision-making."
)