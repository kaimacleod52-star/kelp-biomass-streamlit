# kelp_biomass_predictor.py

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load your data
st.title("🌿 Kelp Biomass Prediction Dashboard")
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preview data
    st.subheader("Raw Data")
    st.write(df.head())

    # Expected columns (check this against your file!)
    expected_cols = ['Latitude', 'Longitude', 'Salinity', 'Depth', 'Temperature', 'Nutrient', 'Biomass']
    if not all(col in df.columns for col in expected_cols):
        st.error(f"Missing required columns. Required: {expected_cols}")
    else:
        # Feature engineering
        X = df[['Latitude', 'Longitude', 'Salinity', 'Depth', 'Temperature', 'Nutrient']]
        y = df['Biomass']

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model creation
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1)  # Regression output
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = model.fit(X_train_scaled, y_train,
                            validation_split=0.2,
                            epochs=200,
                            batch_size=32,
                            callbacks=[early_stop],
                            verbose=0)

        st.success("Model training complete!")

        # Predict on full dataset
        predictions = model.predict(scaler.transform(X))
        df['Predicted Biomass'] = predictions

        # Plot heatmap
        st.subheader("Biomass Prediction Heatmap")
        pivot = df.pivot_table(index='Latitude', columns='Longitude', values='Predicted Biomass')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

        # Prediction form
        st.subheader("Make a Custom Prediction")
        lat = st.number_input("Latitude", value=float(df['Latitude'].mean()))
        lon = st.number_input("Longitude", value=float(df['Longitude'].mean()))
        sal = st.number_input("Salinity", value=float(df['Salinity'].mean()))
        dep = st.number_input("Depth", value=float(df['Depth'].mean()))
        temp = st.number_input("Temperature", value=float(df['Temperature'].mean()))
        nut = st.number_input("Nutrient", value=float(df['Nutrient'].mean()))

        input_data = np.array([[lat, lon, sal, dep, temp, nut]])
        input_scaled = scaler.transform(input_data)
        biomass_pred = model.predict(input_scaled)[0][0]

        st.metric("📈 Predicted Biomass", f"{biomass_pred:.2f}")
