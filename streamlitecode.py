
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "deployment_files/final_xgboost_model.pkl"
FEATURE_PATH = "deployment_files/final_feature_columns.pkl"

# IMPORTANT:
# We use CSV instead of forecast_history.pkl
HISTORY_PATH = "deployment_files/forecast_history.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_history():

    history = pd.read_csv(HISTORY_PATH)

    # Convert date column
    history["Date"] = pd.to_datetime(
        history["Date"],
        errors="coerce"
    )

    # Remove invalid dates
    history = history.dropna(
        subset=["Date"]
    ).copy()

    return history


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

@st.cache_data
def load_features():
    return joblib.load(FEATURE_PATH)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Model file not found:\n\n"
        f"{MODEL_PATH}"
    )

    st.stop()


if not os.path.exists(FEATURE_PATH):

    st.error(
        "❌ Feature file not found:\n\n"
        f"{FEATURE_PATH}"
    )

    st.stop()


if not os.path.exists(HISTORY_PATH):

    st.error(
        "❌ Historical CSV file not found:\n\n"
        f"{HISTORY_PATH}\n\n"
        "Please create forecast_history.csv and place it "
        "inside the deployment_files folder."
    )

    st.stop()


# ============================================================
# LOAD FILES
# ============================================================

try:

    model = load_model()

    history_df = load_history()

    feature_columns = load_features()

except Exception as e:

    st.error(
        "❌ Failed to load deployment files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CHECK REQUIRED HISTORY COLUMNS
# ============================================================

required_history_columns = [
    "Product_ID",
    "Store_ID",
    "Date",
    "Units_Sold",
    "Category",
    "Season",
    "Weather",
    "Sales_Channel",
    "Customer_Segment"
]


missing_history_columns = [
    column
    for column in required_history_columns
    if column not in history_df.columns
]


if missing_history_columns:

    st.error(
        "❌ Historical CSV is missing required columns:"
    )

    st.write(missing_history_columns)

    st.stop()


# ============================================================
# PREPARE HISTORY
# ============================================================

history_df = history_df.sort_values(
    [
        "Product_ID",
        "Store_ID",
        "Date"
    ]
).reset_index(drop=True)


# ============================================================
# HOLIDAY MAP
# ============================================================

holiday_map = {
    pd.Timestamp("2027-01-01"): "New Year"
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sales(
    product_id,
    store_id,
    prediction_date
):

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    prediction_date = pd.Timestamp(
        prediction_date
    )


    # --------------------------------------------------------
    # Get product/store history
    # --------------------------------------------------------

    group_data = history_df[
        (history_df["Product_ID"] == product_id)
        &
        (history_df["Store_ID"] == store_id)
    ].sort_values(
        "Date"
    ).copy()


    # --------------------------------------------------------
    # Check historical data
    # --------------------------------------------------------

    if len(group_data) == 0:

        raise ValueError(
            f"No historical data found for "
            f"{product_id} / {store_id}"
        )


    # --------------------------------------------------------
    # Sales history
    # --------------------------------------------------------

    sales_history = (
        group_data["Units_Sold"]
        .astype(float)
        .tolist()
    )


    # --------------------------------------------------------
    # Need at least 30 observations
    # --------------------------------------------------------

    if len(sales_history) < 30:

        raise ValueError(
            f"Not enough historical data for "
            f"{product_id} / {store_id}. "
            f"At least 30 observations are required."
        )


    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_1 = sales_history[-1]

    lag_7 = sales_history[-7]

    lag_14 = sales_history[-14]


    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    rolling_mean_7 = np.mean(
        sales_history[-7:]
    )

    rolling_mean_14 = np.mean(
        sales_history[-14:]
    )

    rolling_mean_30 = np.mean(
        sales_history[-30:]
    )


    # --------------------------------------------------------
    # Latest business information
    # --------------------------------------------------------

    latest_row = group_data.iloc[-1]


    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    year = prediction_date.year

    month = prediction_date.month

    day = prediction_date.day

    day_of_week = prediction_date.dayofweek

    quarter = prediction_date.quarter

    week_of_year = int(
        prediction_date.isocalendar().week
    )

    is_weekend = int(
        day_of_week >= 5
    )


    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    holiday = holiday_map.get(
        prediction_date,
        "None"
    )


    # ========================================================
    # CREATE PREDICTION ROW
    # ========================================================

    prediction_row = pd.DataFrame([
        {

            "Product_ID": product_id,

            "Store_ID": store_id,

            "Category": latest_row["Category"],

            "Season": latest_row["Season"],

            "Weather": latest_row["Weather"],

            "Sales_Channel": latest_row["Sales_Channel"],

            "Customer_Segment": latest_row["Customer_Segment"],

            "Year": year,

            "Month": month,

            "Day": day,

            "Day_of_Week": day_of_week,

            "Quarter": quarter,

            "Week_of_Year": week_of_year,

            "Is_Weekend": is_weekend,

            "Holiday": holiday,

            "Lag_1": lag_1,

            "Lag_7": lag_7,

            "Lag_14": lag_14,

            "Rolling_Mean_7": rolling_mean_7,

            "Rolling_Mean_14": rolling_mean_14,

            "Rolling_Mean_30": rolling_mean_30

        }
    ])


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    categorical_columns = (
        prediction_row
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )


    prediction_encoded = pd.get_dummies(
        prediction_row,
        columns=categorical_columns,
        drop_first=False
    )


    # ========================================================
    # MATCH TRAINING FEATURES
    # ========================================================

    prediction_encoded = prediction_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        prediction_encoded
    )


    prediction = float(
        np.asarray(prediction)
        .reshape(-1)[0]
    )


    # --------------------------------------------------------
    # Sales cannot be negative
    # --------------------------------------------------------

    prediction = max(
        0,
        prediction
    )


    return round(
        prediction,
        2
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "📈 Sales Forecasting System"
)

st.write(
    "Predict product sales using the trained "
    "Improved XGBoost model."
)


# ============================================================
# DEPLOYMENT STATUS
# ============================================================

st.success(
    "✅ Model and historical data loaded successfully!"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.metric(
        "Model",
        "Improved XGBoost"
    )


with info_col2:

    st.metric(
        "Features",
        len(feature_columns)
    )


with info_col3:

    st.metric(
        "Historical Rows",
        len(history_df)
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔮 Forecast Inputs"
)


# ------------------------------------------------------------
# Products
# ------------------------------------------------------------

products = sorted(
    history_df["Product_ID"]
    .dropna()
    .unique()
    .tolist()
)


# ------------------------------------------------------------
# Stores
# ------------------------------------------------------------

stores = sorted(
    history_df["Store_ID"]
    .dropna()
    .unique()
    .tolist()
)


# ------------------------------------------------------------
# Product selection
# ------------------------------------------------------------

selected_product = st.sidebar.selectbox(
    "Select Product",
    products
)


# ------------------------------------------------------------
# Store selection
# ------------------------------------------------------------

selected_store = st.sidebar.selectbox(
    "Select Store",
    stores
)


# ------------------------------------------------------------
# Forecast date
# ------------------------------------------------------------

selected_date = st.sidebar.date_input(
    "Select Forecast Date",
    value=pd.Timestamp(
        "2027-01-01"
    ).date()
)


# ============================================================
# FORECAST BUTTON
# ============================================================

if st.sidebar.button("🔮 Predict Sales"):

    try:

        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction = predict_sales(
            selected_product,
            selected_store,
            selected_date
        )

        # ----------------------------------------------------
        # Success message
        # ----------------------------------------------------

        st.success(
            "Prediction generated successfully!"
        )

        # ----------------------------------------------------
        # Result cards
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Product",
                selected_product
            )

        with col2:
            st.metric(
                "Store",
                selected_store
            )

        with col3:
            st.metric(
                "Predicted Sales",
                f"{prediction:.2f} units"
            )

        st.divider()

        # ====================================================
        # FORECAST DETAILS
        # ====================================================

        st.subheader(
            "📊 Forecast Details"
        )

        result_df = pd.DataFrame({

            "Product": [
                selected_product
            ],

            "Store": [
                selected_store
            ],

            "Forecast Date": [
                pd.Timestamp(
                    selected_date
                ).strftime("%Y-%m-%d")
            ],

            "Predicted Units Sold": [
                prediction
            ]

        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # INPUT HISTORY INFORMATION
        # ====================================================

        st.subheader(
            "📚 Historical Information Used"
        )

        group_data = history_df[
            (history_df["Product_ID"] == selected_product)
            &
            (history_df["Store_ID"] == selected_store)
        ].sort_values(
            "Date"
        )

        # ----------------------------------------------------
        # Latest historical date
        # ----------------------------------------------------

        latest_date = group_data[
            "Date"
        ].max()

        st.write(
            f"Latest historical date: "
            f"**{latest_date.strftime('%Y-%m-%d')}**"
        )

        # ----------------------------------------------------
        # Number of observations
        # ----------------------------------------------------

        st.write(
            f"Historical observations used: "
            f"**{len(group_data)}**"
        )

        # ----------------------------------------------------
        # Show latest historical records
        # ----------------------------------------------------

        st.dataframe(
            group_data.tail(10),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sales Forecasting System | "
    "Improved XGBoost | "
    "Docker-ready deployment"
)


### One important thing before running this


