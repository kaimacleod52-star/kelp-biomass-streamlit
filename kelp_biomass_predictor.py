# 4kelp_prediction_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Kelp Biomass Prediction Dashboard",
    layout="wide"
)

st.title("🌿 Kelp Biomass Prediction Dashboard")
st.subheader(
    "How can AI-driven analysis of ocean data identify the most sustainable, "
    "high-yield kelp farming locations?"
)

st.markdown(
    """
Upload one or more CSVs (e.g., **CTD_Kelp_combined_by_location.csv**) that include:
- Geographic features (Latitude, Longitude, Depth)
- Ocean conditions (Temperature, Salinity)
- Observed **BIOMASS** values

This app will:
1. Train an AI model (neural network or other regressors) to predict BIOMASS  
2. Map predicted biomass across your study area  
3. Score locations on **sustainability & suitability** for kelp farming
"""
)

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
with st.sidebar:
    st.header("⚙️ Data & Model Settings")

uploaded_files = st.file_uploader(
    "Upload dataset(s) with kelp biomass",
    type=["csv"],
    accept_multiple_files=True
)

# Columns we’ll use from CTD_Kelp_combined_by_location.csv
required_columns = [
    "Latitude",
    "Longitude",
    "Depth",
    "Temperature",
    "Salinity",
    "BIOMASS"      # target
]

if uploaded_files:
    dfs = []
    for uploaded_file in uploaded_files:
        st.sidebar.write(f"📄 **Loaded:** {uploaded_file.name}")
        try:
            temp_df = pd.read_csv(
                uploaded_file,
                sep=",",
                engine="python",
                on_bad_lines="skip"
            )
            dfs.append(temp_df)
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

    if not dfs:
        st.stop()

    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)

    st.success("✅ Data loaded successfully!")
    st.write("**Detected columns:**")
    st.code(", ".join(df.columns))

    # ------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ------------------------------------------------
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(
            "❌ The following required columns are missing from the uploaded file(s): "
            + ", ".join(missing)
        )
        st.info(
            "Please upload a file like CTD_Kelp_combined_by_location.csv "
            "that contains at least: "
            + ", ".join(required_columns)
        )
        st.stop()

    # Keep only needed columns & drop NaNs
    df_model = df[required_columns].copy()
    df_model = df_model.dropna(subset=required_columns)

    st.write("**Rows used for modeling after dropping NaNs:**", len(df_model))

    # ------------------------------------------------
    # QUICK COORDINATE SANITY CHECK
    # ------------------------------------------------
    st.subheader("📍 Coordinate Sanity Check (Raw Data)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("First few rows of coordinates & biomass:")
        st.dataframe(df_model[["Latitude", "Longitude", "Depth", "BIOMASS"]].head())

    with col_b:
        lat_min, lat_max = float(df_model["Latitude"].min()), float(df_model["Latitude"].max())
        lon_min, lon_max = float(df_model["Longitude"].min()), float(df_model["Longitude"].max())
        st.write(f"**Latitude range:** {lat_min:.3f} to {lat_max:.3f}")
        st.write(f"**Longitude range:** {lon_min:.3f} to {lon_max:.3f}")
        st.caption(
            "If these ranges don't match your real-world study area "
            "(e.g., BC coast ~48–55°N, -135 to -120°W), "
            "then the source file likely has incorrect coordinates."
        )

    # Optional: raw sampling locations on a simple map
    try:
        raw_map_df = df_model[["Latitude", "Longitude"]].rename(
            columns={"Latitude": "lat", "Longitude": "lon"}
        )
        st.map(raw_map_df)
    except Exception:
        st.warning("Could not render raw location map. Check coordinate values and types.")

    # ------------------------------------------------
    # FEATURES & TARGET
    # ------------------------------------------------
    feature_cols = ["Latitude", "Longitude", "Depth", "Temperature", "Salinity"]
    target_col = "BIOMASS"

    X = df_model[feature_cols]
    y = df_model[target_col]

    # Optionally log-transform biomass for more stable modeling
    with st.sidebar:
        use_log_target = st.checkbox(
            "Use log-transform on BIOMASS (log(1 + BIOMASS))",
            value=True
        )

    if use_log_target:
        y_transformed = np.log1p(y)
        target_label = "log(1 + BIOMASS)"
    else:
        y_transformed = y
        target_label = "BIOMASS"

    # ------------------------------------------------
    # CORRELATION ANALYSIS
    # ------------------------------------------------
    st.subheader("🔍 Correlation with BIOMASS")

    corr = df_model[feature_cols + [target_col]].corr()
    st.write("Correlation of each feature with `BIOMASS`:")
    corr_with_biomass = corr[target_col].sort_values(ascending=False).to_frame(
        "Correlation with BIOMASS"
    )
    st.dataframe(corr_with_biomass)

    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Matrix")
    st.pyplot(fig_corr)

    # ------------------------------------------------
    # TRAIN / TEST SPLIT & SCALING
    # ------------------------------------------------
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_transformed, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------
    # MODEL SELECTION & TRAINING
    # ------------------------------------------------
    st.markdown("### 🧠 Model Selection & Training")

    with st.sidebar:
        model_name = st.selectbox(
            "Choose model type:",
            [
                "Neural Network (MLPRegressor)",
                "Random Forest",
                "Linear Regression"
            ]
        )

    if model_name == "Neural Network (MLPRegressor)":
        st.info("You selected a **Neural Network** (MLPRegressor).")

        with st.sidebar:
            hidden_layer_sizes = st.slider(
                "Hidden layer sizes (first, second)",
                min_value=8,
                max_value=256,
                value=(128, 64),
                step=8
            )
            max_iter = st.slider(
                "Maximum training iterations",
                min_value=200,
                max_value=3000,
                value=1000,
                step=100
            )
            alpha = st.select_slider(
                "L2 regularization (alpha)",
                options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                value=1e-3
            )
            learning_rate_init = st.select_slider(
                "Initial learning rate",
                options=[0.0001, 0.001, 0.01],
                value=0.001
            )

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            random_state=42
        )
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled

    elif model_name == "Random Forest":
        st.info("You selected a **Random Forest** regressor.")

        with st.sidebar:
            n_estimators = st.slider(
                "Number of trees (n_estimators)",
                min_value=50,
                max_value=600,
                value=300,
                step=50
            )
            max_depth = st.slider(
                "Maximum depth of trees (max_depth)",
                min_value=2,
                max_value=40,
                value=18,
                step=2
            )
            min_samples_leaf = st.slider(
                "Min samples per leaf",
                min_value=1,
                max_value=20,
                value=3,
                step=1
            )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled

    else:  # Linear Regression
        st.info("You selected a **Linear Regression** model.")
        model = LinearRegression()
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled

    # Fit model
    model.fit(X_train_used, y_train_raw)
    y_pred_transformed = model.predict(X_test_used)

    # Convert predictions back to BIOMASS scale for interpretation
    if use_log_target:
        y_test = np.expm1(y_test_raw)
        y_pred = np.expm1(y_pred_transformed)
    else:
        y_test = y_test_raw
        y_pred = y_pred_transformed

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.success("✅ Model training complete!")
    st.write(f"**Model:** {model_name}")
    st.write(f"**RMSE (test set, BIOMASS units):** {rmse:.3f}")
    st.write(f"**R² (test set):** {r2:.3f}")

    # Loss curve for NN
    if model_name == "Neural Network (MLPRegressor)" and hasattr(model, "loss_curve_"):
        st.subheader("📉 Neural Network Training Loss")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(model.loss_curve_)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Loss Curve")
        st.pyplot(fig_loss)

    # ------------------------------------------------
    # ACTUAL VS PREDICTED PLOT
    # ------------------------------------------------
    st.subheader("📊 Actual vs Predicted BIOMASS")

    fig_avp, ax_avp = plt.subplots()
    ax_avp.scatter(y_test, y_pred, alpha=0.6)
    line_min = min(y_test.min(), y_pred.min())
    line_max = max(y_test.max(), y_pred.max())
    ax_avp.plot([line_min, line_max], [line_min, line_max], "r--")
    ax_avp.set_xlabel("Actual BIOMASS")
    ax_avp.set_ylabel("Predicted BIOMASS")
    ax_avp.set_title(f"Actual vs Predicted ({model_name})")
    st.pyplot(fig_avp)

    # ------------------------------------------------
    # FEATURE IMPORTANCE (Permutation Importance)
    # ------------------------------------------------
    st.subheader("📌 Feature Importance (Permutation)")

    with st.spinner("Computing permutation importance (this may take a moment)..."):
        perm = permutation_importance(
            model,
            X_test_used,
            y_test_raw,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance (mean decrease in score)": perm.importances_mean
    }).sort_values(
        "Importance (mean decrease in score)",
        ascending=False
    )

    st.dataframe(importances)

    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(
        importances["Feature"],
        importances["Importance (mean decrease in score)"]
    )
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("Permutation-based Feature Importance")
    st.pyplot(fig_imp)

    # ------------------------------------------------
    # BIOMASS PREDICTION MAPS
    # ------------------------------------------------
    st.subheader("🌍 Biomass Prediction Map")

    # Predict on all rows for mapping
    X_all_scaled = scaler.transform(df_model[feature_cols])
    preds_all_transformed = model.predict(X_all_scaled)

    if use_log_target:
        preds_all = np.expm1(preds_all_transformed)
    else:
        preds_all = preds_all_transformed

    preds_all = np.maximum(preds_all, 0)  # no negative biomass

    map_df = df_model.copy()
    map_df["Predicted_BIOMASS"] = preds_all

    # Map style choice
    map_style = st.radio(
        "Map style",
        ["Point map (static)", "Binned heatmap (static)", "Interactive map (OpenStreetMap)"],
        index=2,
        horizontal=True
    )

    if map_style == "Point map (static)":
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#f0f0f0")
        sc = ax.scatter(
            map_df["Longitude"],
            map_df["Latitude"],
            c=map_df["Predicted_BIOMASS"],
            s=15,
            cmap="viridis"
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Predicted BIOMASS")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Predicted Kelp Biomass (point-level)")
        ax.set_aspect("equal", adjustable="box")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif map_style == "Binned heatmap (static)":
        bin_decimals = st.slider(
            "Heatmap resolution (fewer decimals = smoother map)",
            min_value=1,
            max_value=4,
            value=2,
            step=1
        )

        map_df["lat_bin"] = map_df["Latitude"].round(bin_decimals)
        map_df["lon_bin"] = map_df["Longitude"].round(bin_decimals)

        grouped = (
            map_df
            .groupby(["lat_bin", "lon_bin"])["Predicted_BIOMASS"]
            .mean()
            .reset_index()
        )

        heatmap_data = grouped.pivot(
            index="lat_bin",
            columns="lon_bin",
            values="Predicted_BIOMASS"
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor("#f0f0f0")
        hm = sns.heatmap(
            heatmap_data,
            cmap="viridis",
            ax=ax,
            cbar=True
        )
        cbar = hm.collections[0].colorbar
        cbar.set_label("Predicted BIOMASS")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Average Predicted Kelp Biomass (binned)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Predicted_BIOMASS",
            color_continuous_scale="Viridis",
            size="Predicted_BIOMASS",
            size_max=15,
            zoom=5,
            height=600,
            hover_data={
                "Predicted_BIOMASS": True,
                "Depth": True,
                "Temperature": True,
                "Salinity": True,
            },
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=30, b=0),
            title="Predicted Kelp Biomass – Interactive Map",
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # ------------------------------------------------
    # SUSTAINABILITY & SITE SUITABILITY
    # ------------------------------------------------
    st.subheader("♻️ Sustainable & High-Yield Site Selection")

    st.markdown(
        """
Use the controls below to define environmentally preferred ranges for kelp farming.
The app will:
- Filter locations that meet your constraints
- Compute a **Site Suitability Score** = blend of predicted biomass and environmental fit
- Show the **top candidate locations** for sustainable, high-yield farming
"""
    )

    col_depth, col_temp, col_sal = st.columns(3)

    with col_depth:
        st.markdown("**Preferred Depth (m)**")
        depth_min = st.number_input("Min depth", value=5.0, step=1.0, key="depth_min")
        depth_max = st.number_input("Max depth", value=40.0, step=1.0, key="depth_max")

    with col_temp:
        st.markdown("**Preferred Temperature (°C)**")
        temp_min = st.number_input("Min temp", value=5.0, step=0.5, key="temp_min")
        temp_max = st.number_input("Max temp", value=18.0, step=0.5, key="temp_max")

    with col_sal:
        st.markdown("**Preferred Salinity (PSU)**")
        sal_min = st.number_input("Min salinity", value=28.0, step=0.5, key="sal_min")
        sal_max = st.number_input("Max salinity", value=35.0, step=0.5, key="sal_max")

    col_weights1, col_weights2 = st.columns(2)
    with col_weights1:
        biomass_weight = st.slider(
            "Weight on biomass (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
    with col_weights2:
        env_weight = 1.0 - biomass_weight
        st.write(f"Weight on environmental suitability: **{env_weight:.2f}**")

    # Filter by hard constraints
    sustainable_df = map_df[
        (map_df["Depth"].between(depth_min, depth_max))
        & (map_df["Temperature"].between(temp_min, temp_max))
        & (map_df["Salinity"].between(sal_min, sal_max))
    ].copy()

    if sustainable_df.empty:
        st.warning("No locations meet the current sustainability constraints. Try widening the ranges.")
    else:
        # Normalize biomass and environmental 'fit' scores
        # Biomass score: normalized 0-1
        biomass_vals = sustainable_df["Predicted_BIOMASS"].values
        if biomass_vals.max() - biomass_vals.min() > 0:
            biomass_score = (biomass_vals - biomass_vals.min()) / (biomass_vals.max() - biomass_vals.min())
        else:
            biomass_score = np.ones_like(biomass_vals) * 0.5  # constant biomass case

        # Environmental score: closer to mid-point of preferred range = better
        def closeness_score(value, vmin, vmax):
            mid = 0.5 * (vmin + vmax)
            # distance from midpoint normalized by half-range
            half_range = max((vmax - vmin) / 2.0, 1e-6)
            dist = np.abs(value - mid) / half_range
            # convert to score between 0 and 1 (1 = ideal)
            return np.clip(1 - dist, 0, 1)

        depth_score = closeness_score(sustainable_df["Depth"].values, depth_min, depth_max)
        temp_score = closeness_score(sustainable_df["Temperature"].values, temp_min, temp_max)
        sal_score = closeness_score(sustainable_df["Salinity"].values, sal_min, sal_max)

        env_score = (depth_score + temp_score + sal_score) / 3.0

        # Combined suitability score
        suitability_score = biomass_weight * biomass_score + env_weight * env_score
        sustainable_df["Suitability_Score"] = suitability_score

        # Show top candidate sites
        top_n = st.slider(
            "Number of top candidate sites to display",
            min_value=5,
            max_value=min(100, len(sustainable_df)),
            value=min(20, len(sustainable_df)),
            step=1
        )

        top_sites = sustainable_df.sort_values(
            "Suitability_Score", ascending=False
        ).head(top_n)

        st.markdown("**Top Candidate Sites (ranked by Suitability Score):**")
        st.dataframe(
            top_sites[[
                "Latitude",
                "Longitude",
                "Depth",
                "Temperature",
                "Salinity",
                "Predicted_BIOMASS",
                "Suitability_Score"
            ]]
        )

        # Optional: plot top sites on interactive map
        st.markdown("**Top Sites on Interactive Map:**")
        fig_top = px.scatter_mapbox(
            top_sites,
            lat="Latitude",
            lon="Longitude",
            color="Suitability_Score",
            color_continuous_scale="Turbo",
            size="Predicted_BIOMASS",
            size_max=18,
            zoom=5,
            height=600,
            hover_data={
                "Predicted_BIOMASS": True,
                "Suitability_Score": True,
                "Depth": True,
                "Temperature": True,
                "Salinity": True,
            },
        )
        fig_top.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=30, b=0),
            title="Top Sustainable, High-Yield Candidate Sites",
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # ------------------------------------------------
    # INTERACTIVE SINGLE-POINT PREDICTOR
    # ------------------------------------------------
    st.markdown("### 🎯 Predict Biomass for Custom Conditions")

    st.caption(
        "Experiment with hypothetical locations and conditions to see how the model predicts BIOMASS."
    )

    input_data = {}
    for feature in feature_cols:
        try:
            default_val = float(df_model[feature].mean())
        except Exception:
            default_val = 0.0
        input_data[feature] = st.number_input(
            feature,
            value=default_val,
            step=0.01
        )

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    input_pred_transformed = model.predict(input_scaled)[0]

    if use_log_target:
        predicted_biomass_single = np.expm1(input_pred_transformed)
    else:
        predicted_biomass_single = input_pred_transformed

    predicted_biomass_single = max(predicted_biomass_single, 0)

    st.metric("📈 Predicted Biomass", round(predicted_biomass_single, 3))

else:
    st.info("Awaiting CSV file upload…")
    st.markdown(
        "Upload **CTD_Kelp_combined_by_location.csv** or a similar file that "
        "contains at least these columns:\n\n"
        + ", ".join(required_columns)
    )

