# kelp_prediction_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import io

st.set_page_config(page_title="Kelp Biomass Prediction Dashboard")
st.title("🌿 Kelp Biomass Prediction Dashboard")
st.subheader("Upload dataset(s)")

uploaded_files = st.file_uploader(
    "Upload dataset(s)", type=["csv"], accept_multiple_files=True
)

required_columns = [
    'Latitude', 'Longitude', 'Salinity', 'Depth', 'Temperature',
    'Nutrient', 'Oxygen', 'Phosphate', 'Silicate', 'Nitrate and Nitrite+Nitrite',
    'pH', 'Chlorophyll', 'Alkalinity', 'Dissolved Inorganic Carbon',
    'Transmissivity', 'Biomass'
]

if uploaded_files:
    dfs = []  # list to store individual DataFrames
    for uploaded_file in uploaded_files:
        st.write(f"📄 **File:** {uploaded_file.name}")
        try:
            temp_df = pd.read_csv(uploaded_file, sep=",", engine="python", on_bad_lines='skip')
            dfs.append(temp_df)
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)            
    st.info("Please upload a dataset with the following columns:")
    st.code(', '.join(required_columns))

    # Drop rows with missing values
    df.dropna(inplace=True)
    features = [
        "Latitude", "Longitude", "Salinity", "Depth", "Temperature", "Nutrient",
        "Oxygen", "Phosphate", "Silicate", "Nitrate and Nitrite+Nitrite", "pH", "Chlorophyll",
        "Alkalinity", "Dissolved_Inorganic_Carbon", "Transmissivity"
    ]
    target = "Biomass"

    # Split and scale
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    st.success("Model training complete!")


    # Create a new DataFrame for heatmap
    predict_features = [
    "Latitude", "Longitude", "Salinity", "Depth", "Temperature", "Nutrient",
    "Oxygen", "Phosphate", "Silicate", "Nitrate and Nitrite+Nitrite", "pH", "Chlorophyll",
    "Alkalinity", "Dissolved_Inorganic_Carbon", "Transmissivity"
    ]
    X_scaled = scaler.transform(df[predict_features])
    predictions = model.predict(X_scaled)
    heatmap_df = df.copy()
    heatmap_df["Predicted_Biomass"] = predictions

    heatmap_df["lat_bin"] = heatmap_df["Latitude"].round(2)
    heatmap_df["lon_bin"] = heatmap_df["Longitude"].round(2)

    # Average predicted biomass per bin
    grouped = (
        heatmap_df.groupby(["lat_bin", "lon_bin"])["Predicted_Biomass"]
        .mean()
        .reset_index()
    )

    # Pivot to 2D grid
    heatmap_data = grouped.pivot(index="lat_bin", columns="lon_bin", values="Predicted_Biomass")

    # Plot
    st.subheader("🌍 Biomass Prediction Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax, cbar=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)


    # Prediction Interface
    st.write("### Predict Biomass for Custom Values")
    input_data = {}
    for feature in features:
        try:
            default_val = float(df[feature].mean())
        except:
            default_val = 0.0  # fallback if mean fails
        input_data[feature] = st.number_input(feature, value=default_val, step=0.01)
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    predicted_biomass = model.predict(input_scaled)[0]
    st.metric("📈 Predicted Biomass", round(predicted_biomass, 2))
else:
    st.info("Awaiting CSV file upload...")
    st.markdown("Please upload a dataset with the following columns:")
    st.code(", ".join([
        "Latitude", "Longitude", "Salinity", "Depth", "Temperature", "Nutrient",
        "Oxygen", "Phosphate", "Silicate", "Nitrate and Nitrite+Nitrite", "pH", "Chlorophyll",
        "Alkalinity", "Dissolved_Inorganic_Carbon", "Transmissivity", "Biomass"
    ]))
