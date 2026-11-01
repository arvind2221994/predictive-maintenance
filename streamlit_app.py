from pathlib import Path
from pickle import UnpicklingError
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

st.set_page_config(page_title="Engine Predictive Maintenance", layout="centered")
st.title("Engine Predictive Maintenance")
st.caption("Enter sensor readings to use the tuned model saved by the notebook.")

MODEL_PATH = Path(__file__).resolve().parent.joinpath("models", "best_engine_model.pkl")
INPUT_SETTINGS = {
    "Engine rpm": {"label": "Engine speed (RPM)", "value": 900.0, "step": 10.0, "min_value": 0.0},
    "Lub oil pressure": {"label": "Lubricating oil pressure (bar)", "value": 3.0, "step": 0.1, "min_value": 0.0},
    "Fuel pressure": {"label": "Fuel pressure (bar)", "value": 6.0, "step": 0.1, "min_value": 0.0},
    "Coolant pressure": {"label": "Coolant pressure (bar)", "value": 2.0, "step": 0.1, "min_value": 0.0},
    "lub oil temp": {"label": "Lubricating oil temperature (degrees C)", "value": 78.0, "step": 0.1},
    "Coolant temp": {"label": "Coolant temperature (degrees C)", "value": 80.0, "step": 0.1},
}


@st.cache_resource
def load_model(model_file: str, model_revision: int):
    # The revision participates in Streamlit's cache key when CI replaces the model.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(model_file)


try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Missing models/best_engine_model.pkl. Commit the model saved by the notebook."
        )
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except (OSError, ImportError, ValueError, EOFError, UnpicklingError, InconsistentVersionWarning) as error:
    st.error(f"Unable to load the committed model: {error}")
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error("The saved model must be fitted on a DataFrame with named sensor columns.")
    st.stop()

feature_columns = list(model.feature_names_in_)
if len(feature_columns) != len(INPUT_SETTINGS) or set(feature_columns) != set(INPUT_SETTINGS):
    st.error("The committed model's input schema does not match the six raw sensor inputs.")
    st.stop()

with st.form("engine_sensor_form"):
    columns = st.columns(2)
    sensor_inputs = {}
    for index, (feature, settings) in enumerate(INPUT_SETTINGS.items()):
        with columns[index % 2]:
            sensor_inputs[feature] = st.number_input(key=feature, **settings)
    submitted = st.form_submit_button("Predict engine condition")

if submitted:
    input_df = pd.DataFrame([sensor_inputs], columns=feature_columns)
    if not np.isfinite(input_df.to_numpy(dtype=float)).all():
        st.error("All sensor readings must be finite numbers.")
        st.stop()

    try:
        predicted_condition = model.predict(input_df)[0]
        class_probabilities = model.predict_proba(input_df)[0]
    except (ValueError, TypeError) as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    st.session_state["input_df"] = input_df
    st.session_state["predicted_condition"] = str(predicted_condition)
    st.session_state["class_probabilities"] = pd.DataFrame({
        "Engine Condition": [str(label) for label in model.classes_],
        "Probability": class_probabilities,
    })

if "input_df" in st.session_state:
    st.subheader("Latest submitted sensor readings")
    st.dataframe(st.session_state["input_df"], hide_index=True)
    st.metric("Predicted Engine Condition", st.session_state["predicted_condition"])
    st.subheader("Model probabilities by class")
    st.dataframe(st.session_state["class_probabilities"], hide_index=True)
    st.caption(
        "Confirm the dataset's class-label mapping before interpreting a label as healthy or faulty. "
        "Model scores are not calibrated failure-risk estimates or automatic maintenance instructions."
    )
    st.download_button(
        "Download submitted inputs (CSV)",
        data=st.session_state["input_df"].to_csv(index=False).encode("utf-8"),
        file_name="engine_inputs.csv",
        mime="text/csv",
    )
