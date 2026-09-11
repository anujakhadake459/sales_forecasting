import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date

st.markdown("""
<style>
/* Main text */
html, body, [class*="css"] {
    font-size: 14px;
}

/* Sidebar text */
section[data-testid="stSidebar"] {
    font-size: 13px;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    font-size: 13px !important;
}

/* Headings */
h1 {
    font-size: 28px !important;
}

h2 {
    font-size: 22px !important;
}

h3 {
    font-size: 18px !important;
}

/* Buttons */
.stButton > button {
    font-size: 14px !important;
}

/* Input fields */
.stSelectbox label,
.stNumberInput label,
.stDateInput label,
.stSlider label {
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "deployment_files/final_xgboost_model.pkl"
FEATURE_PATH = "deployment_files/final_feature_columns.pkl"

# Original dataset
DATASET_PATH = "Sales_Forcasting_Dataset_updated.xlsx"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_excel(DATASET_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    ).copy()

    return df


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

@st.cache_data
def load_features():

    return joblib.load(
        FEATURE_PATH
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"""
        ❌ Model file not found.

        Expected:
        {MODEL_PATH}
        """
    )

    st.stop()


if not os.path.exists(FEATURE_PATH):

    st.error(
        f"""
        ❌ Feature file not found.

        Expected:
        {FEATURE_PATH}
        """
    )

    st.stop()


if not os.path.exists(DATASET_PATH):

    st.error(
        f"""
        ❌ Original dataset not found.

        Expected:
        {DATASET_PATH}

        Please keep Sales_Forcasting_Dataset_updated.xlsx
        in the same folder as app.py.
        """
    )

    st.stop()


# ============================================================
# LOAD FILES
# ============================================================

try:

    model = load_model()

    dataset_df = load_dataset()

    feature_columns = load_features()

except Exception as e:

    st.error(
        "❌ Failed to load deployment files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# REQUIRED DATASET COLUMNS
# ============================================================

required_columns = [

    "Date",
    "Product_ID",
    "Product_Name",
    "Category",
    "Store_ID",
    "Store_Location",
    "Units_Sold",
    "Price",
    "Discount_Percentage",
    "Promotion_Flag",
    "Stock_Availability",
    "Day_of_Week",
    "Month",
    "Quarter",
    "Holiday_Flag",
    "Is_Weekend",
    "Season",
    "Weather",
    "Local_Event_Flag",
    "Competitor_Price",
    "Economic_Indicator",
    "Sales_Channel",
    "Customer_Segment",
    "Marketing_Spend"

]


missing_columns = [

    col
    for col in required_columns
    if col not in dataset_df.columns

]


if missing_columns:

    st.error(
        "❌ Required columns are missing from the original dataset:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

dataset_df = dataset_df.sort_values(
    [
        "Product_ID",
        "Store_ID",
        "Date"
    ]
).reset_index(drop=True)


# ============================================================
# CREATE MODEL HISTORY
#
# IMPORTANT:
# Product_Name and Store_Location are NOT model features.
# They are only used for displaying names in the application.
# ============================================================

model_history_df = dataset_df.drop(
    columns=[
        "Row_ID",
        "Revenue",
        "Product_Name",
        "Store_Location",
        "Holiday_Name"
    ],
    errors="ignore"
).copy()


# ============================================================
# CREATE HOLIDAY LOOKUP
#
# Use the original dataset values.
# ============================================================

holiday_lookup = set(

    zip(
        dataset_df.loc[
            dataset_df["Holiday_Flag"] == 1,
            "Date"
        ].dt.month,

        dataset_df.loc[
            dataset_df["Holiday_Flag"] == 1,
            "Date"
        ].dt.day
    )

)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_product_name(product_id):

    rows = dataset_df[
        dataset_df["Product_ID"] == product_id
    ]

    if len(rows) == 0:
        return str(product_id)

    return str(
        rows.iloc[0]["Product_Name"]
    )


def get_store_name(store_id):

    rows = dataset_df[
        dataset_df["Store_ID"] == store_id
    ]

    if len(rows) == 0:
        return str(store_id)

    return str(
        rows.iloc[0]["Store_Location"]
    )


def product_display_name(product_id):

    return (
        f"{product_id} - "
        f"{get_product_name(product_id)}"
    )


def store_display_name(store_id):

    return (
        f"{store_id} - "
        f"{get_store_name(store_id)}"
    )


# ============================================================
# SEASON FUNCTION
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:

        return "Winter"

    elif month in [3, 4, 5]:

        return "Spring"

    elif month in [6, 7, 8]:

        return "Summer"

    else:

        return "Autumn"


# ============================================================
# HOLIDAY FLAG
# ============================================================

def get_future_holiday_flag(prediction_date):

    key = (
        prediction_date.month,
        prediction_date.day
    )

    if key in holiday_lookup:

        return 1

    return 0


# ============================================================
# PREPARE MODEL HISTORY
# ============================================================

model_history_df = model_history_df.sort_values(
    [
        "Product_ID",
        "Store_ID",
        "Date"
    ]
).reset_index(drop=True)


# ============================================================
# PREDICTION FUNCTION
#
# IMPORTANT:
# This recreates the same feature structure used by the
# XGBoost model.
# ============================================================

def predict_sales(
    product_id,
    store_id,
    prediction_date,
    working_history,
    user_inputs=None
):

    prediction_date = pd.Timestamp(
        prediction_date
    )


    # --------------------------------------------------------
    # Product + Store history
    # --------------------------------------------------------

    history = working_history[
        (working_history["Product_ID"] == product_id)
        &
        (working_history["Store_ID"] == store_id)
    ].copy()


    history = history.sort_values(
        "Date"
    ).reset_index(drop=True)


    if len(history) == 0:

        raise ValueError(
            f"No historical data found for "
            f"{product_id} + {store_id}"
        )


    # --------------------------------------------------------
    # Sales history
    # --------------------------------------------------------

    sales_history = (

        history["Units_Sold"]
        .astype(float)
        .tolist()

    )


    if len(sales_history) < 30:

        raise ValueError(
            f"""
            Not enough historical records for
            {product_id} + {store_id}.

            At least 30 observations are required.
            Available observations: {len(sales_history)}
            """
        )


    # --------------------------------------------------------
    # Lag features
    #
    # Same logic used in notebook:
    #
    # Lag_1
    # Lag_7
    # Lag_14
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
    # Latest row / user-selected business inputs
    # --------------------------------------------------------

    latest_row = history.iloc[-1]

    # When the deployment user supplies business inputs, use
    # those values for the forecast. Otherwise keep the latest
    # known historical values for backward compatibility.
    inputs = user_inputs or {}

    price_value = inputs.get("Price", latest_row["Price"])
    discount_value = inputs.get("Discount_Percentage", latest_row["Discount_Percentage"])
    promotion_value = inputs.get("Promotion_Flag", latest_row["Promotion_Flag"])
    stock_value = inputs.get("Stock_Availability", latest_row["Stock_Availability"])
    weather_value = inputs.get("Weather", latest_row["Weather"])
    local_event_value = inputs.get("Local_Event_Flag", latest_row["Local_Event_Flag"])
    competitor_value = inputs.get("Competitor_Price", latest_row["Competitor_Price"])
    economic_value = inputs.get("Economic_Indicator", latest_row["Economic_Indicator"])
    channel_value = inputs.get("Sales_Channel", latest_row["Sales_Channel"])
    segment_value = inputs.get("Customer_Segment", latest_row["Customer_Segment"])


    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    day_of_week = prediction_date.day_name()

    month_name = prediction_date.month_name()

    quarter = (
        "Q"
        + str(prediction_date.quarter)
    )

    is_weekend = int(
        prediction_date.dayofweek >= 5
    )

    season = get_season(
        prediction_date.month
    )


    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    holiday_flag = inputs.get(
        "Holiday_Flag",
        get_future_holiday_flag(prediction_date)
    )


    # ========================================================
    # CREATE MODEL INPUT
    #
    # IMPORTANT:
    # These are the same raw model features used during
    # training.
    #
    # Product_Name and Store_Location are intentionally
    # NOT included because the notebook removed them before
    # XGBoost training.
    # ========================================================

    prediction_row = pd.DataFrame({

        "Product_ID": [
            product_id
        ],

        "Category": [
            latest_row["Category"]
        ],

        "Store_ID": [
            store_id
        ],

        "Price": [
            price_value
        ],

        "Discount_Percentage": [
            discount_value
        ],

        "Promotion_Flag": [
            promotion_value
        ],

        "Stock_Availability": [
            stock_value
        ],

        "Day_of_Week": [
            day_of_week
        ],

        "Month": [
            month_name
        ],

        "Quarter": [
            quarter
        ],

        "Holiday_Flag": [
            holiday_flag
        ],

        "Is_Weekend": [
            is_weekend
        ],

        "Season": [
            season
        ],

        "Weather": [
            weather_value
        ],

        "Local_Event_Flag": [
            local_event_value
        ],

        "Competitor_Price": [
            competitor_value
        ],

        "Economic_Indicator": [
            economic_value
        ],

        "Sales_Channel": [
            channel_value
        ],

        "Customer_Segment": [
            segment_value
        ],

        "Marketing_Spend": [
            latest_row["Marketing_Spend"]
        ],

        "Lag_1": [
            lag_1
        ],

        "Lag_7": [
            lag_7
        ],

        "Lag_14": [
            lag_14
        ],

        "Rolling_Mean_7": [
            rolling_mean_7
        ],

        "Rolling_Mean_14": [
            rolling_mean_14
        ],

        "Rolling_Mean_30": [
            rolling_mean_30
        ]

    })


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    categorical_columns = (

        prediction_row
        .select_dtypes(
            include=["object"]
        )
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

    prediction_encoded = (

        prediction_encoded
        .reindex(
            columns=feature_columns,
            fill_value=0
        )

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
    # No negative sales
    # --------------------------------------------------------

    prediction = max(
        0,
        prediction
    )


    # --------------------------------------------------------
    # Whole number prediction
    #
    # Example:
    # 59.42 -> 59
    # 59.78 -> 60
    # --------------------------------------------------------

    prediction = int(
        round(prediction)
    )


    return prediction


# ============================================================
# RECURSIVE FORECAST
#
# Generates 1 / 7 / 14 / 30 days.
# Each predicted value is added to the working history so
# the next day's lag features can use the previous prediction.
# ============================================================

def generate_forecast(
    product_id,
    store_id,
    horizon,
    base_history,
    start_date=None,
    user_inputs=None
):

    history = base_history.copy()

    history = history.sort_values(
        "Date"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Latest date for this product + store
    # --------------------------------------------------------

    pair_history = history[
        (history["Product_ID"] == product_id)
        &
        (history["Store_ID"] == store_id)
    ].copy()


    if len(pair_history) == 0:

        raise ValueError(
            f"No history found for "
            f"{product_id} + {store_id}"
        )


    latest_date = pair_history["Date"].max()


    # --------------------------------------------------------
    # Forecast starts from next day
    # --------------------------------------------------------

    if start_date is None:
        forecast_start = latest_date + pd.Timedelta(days=1)
    else:
        forecast_start = pd.Timestamp(start_date)
        if forecast_start <= latest_date:
            raise ValueError(
                f"Forecast start date must be after the latest "
                f"historical date ({latest_date.strftime('%Y-%m-%d')})."
            )


    forecast_rows = []


    # --------------------------------------------------------
    # Generate forecast day by day
    # --------------------------------------------------------

    for i in range(horizon):

        forecast_date = (
            forecast_start
            + pd.Timedelta(days=i)
        )


        prediction = predict_sales(

            product_id,

            store_id,

            forecast_date,

            history,

            user_inputs

        )


        forecast_rows.append({

            "Date": forecast_date,

            "Product_ID": product_id,

            "Store_ID": store_id,

            "Predicted_Units_Sold": prediction,
            "Forecast_Start_Date": forecast_start

        })


        # ----------------------------------------------------
        # Add predicted value into history
        #
        # This is required for recursive forecasting.
        # ----------------------------------------------------

        latest_pair_row = history[
            (history["Product_ID"] == product_id)
            &
            (history["Store_ID"] == store_id)
        ].sort_values(
            "Date"
        ).iloc[-1].copy()


        latest_pair_row["Date"] = forecast_date

        latest_pair_row["Units_Sold"] = prediction

        latest_pair_row["Day_of_Week"] = (
            forecast_date.day_name()
        )

        latest_pair_row["Month"] = (
            forecast_date.month_name()
        )

        latest_pair_row["Quarter"] = (
            "Q"
            + str(forecast_date.quarter)
        )

        latest_pair_row["Holiday_Flag"] = (
            get_future_holiday_flag(
                forecast_date
            )
        )

        latest_pair_row["Is_Weekend"] = int(
            forecast_date.dayofweek >= 5
        )

        latest_pair_row["Season"] = (
            get_season(
                forecast_date.month
            )
        )

        # Keep the selected business inputs in the recursive
        # history so they are used on the next forecast day too.
        if user_inputs:
            for key, value in user_inputs.items():
                if key in latest_pair_row.index:
                    latest_pair_row[key] = value


        history = pd.concat(

            [
                history,
                pd.DataFrame(
                    [latest_pair_row]
                )
            ],

            ignore_index=True

        )


    forecast_df = pd.DataFrame(
        forecast_rows
    )


    return forecast_df


# ============================================================
# STORE TOTAL FORECAST
# ============================================================

def generate_store_total_forecast(
    store_id,
    horizon,
    base_history,
    start_date=None,
    user_inputs=None
):

    store_history = base_history[
        base_history["Store_ID"] == store_id
    ].copy()


    products_for_store = (

        store_history["Product_ID"]
        .dropna()
        .unique()
        .tolist()

    )


    all_product_forecasts = []

    skipped_products = []


    for product_id in products_for_store:

        pair_count = len(
            store_history[
                store_history["Product_ID"] == product_id
            ]
        )


        if pair_count < 30:

            skipped_products.append(
                product_id
            )

            continue


        try:

            product_forecast = generate_forecast(

                product_id,

                store_id,

                horizon,

                base_history,

                start_date,

                user_inputs

            )


            all_product_forecasts.append(
                product_forecast
            )


        except Exception:

            skipped_products.append(
                product_id
            )


    if len(all_product_forecasts) == 0:

        raise ValueError(
            f"No products have enough historical "
            f"data for store {store_id}."
        )


    combined = pd.concat(
        all_product_forecasts,
        ignore_index=True
    )


    store_total = (

        combined
        .groupby("Date", as_index=False)
        ["Predicted_Units_Sold"]
        .sum()

    )


    store_total["Predicted_Units_Sold"] = (

        store_total["Predicted_Units_Sold"]
        .round()
        .astype(int)

    )


    return (
        store_total,
        combined,
        skipped_products
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "📈 Sales Forecasting System"
)

st.write(
    "Daily sales forecasting using the trained "
    "Improved XGBoost model."
)


# ============================================================
# DEPLOYMENT STATUS
# ============================================================

st.success(
    "✅ Model and original dataset loaded successfully!"
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
        "Model Features",
        len(feature_columns)
    )


with info_col3:

    st.metric(
        "Dataset Rows",
        len(dataset_df)
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔮 Forecast Inputs"
)


# ============================================================
# FORECAST TYPE
# ============================================================

forecast_type = st.sidebar.radio(
    "Forecast Type",
    [
        "Product + Store",
        "Store Total"
    ]
)


# ============================================================
# INPUTS FOR PRODUCT + STORE FORECAST
# ============================================================

if forecast_type == "Product + Store":

    # --------------------------------------------------------
    # REGION / STORE INPUT
    # --------------------------------------------------------

    region_options = (
        dataset_df["Store_Location"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_region = st.sidebar.selectbox(
        "🌍 Region",
        region_options
    )

    region_store_df = dataset_df[
        dataset_df["Store_Location"].astype(str) == str(selected_region)
    ][["Store_ID", "Store_Location"]].drop_duplicates().sort_values("Store_ID")

    store_display_options = [
        f"{row['Store_ID']} - {row['Store_Location']}"
        for _, row in region_store_df.iterrows()
    ]

    selected_store_display = st.sidebar.selectbox(
        "🏬 Store / Location",
        store_display_options
    )

    selected_store = selected_store_display.split(" - ", 1)[0]


    # --------------------------------------------------------
    # PRODUCT INPUT
    # --------------------------------------------------------

    product_options_df = (
        dataset_df[["Product_ID", "Product_Name"]]
        .drop_duplicates()
        .sort_values("Product_ID")
    )

    product_display_options = [
        f"{row['Product_ID']} - {row['Product_Name']}"
        for _, row in product_options_df.iterrows()
    ]

    selected_product_display = st.sidebar.selectbox(
        "📦 Product",
        product_display_options
    )

    selected_product = selected_product_display.split(" - ", 1)[0]


    # --------------------------------------------------------
    # DEFAULT BUSINESS VALUES
    # --------------------------------------------------------

    selected_pair = dataset_df[
        (dataset_df["Product_ID"] == selected_product)
        & (dataset_df["Store_ID"] == selected_store)
    ].sort_values("Date")

    if len(selected_pair) == 0:
        selected_pair = dataset_df[
            dataset_df["Store_ID"] == selected_store
        ].sort_values("Date")

    latest_input_row = selected_pair.iloc[-1]


    # --------------------------------------------------------
    # BUSINESS FEATURE INPUTS
    # --------------------------------------------------------

    st.sidebar.markdown("### ⚙️ Business Features")

    price_input = st.sidebar.number_input(
        "💰 Price",
        min_value=0.0,
        value=float(latest_input_row["Price"]),
        step=1.0
    )

    discount_input = st.sidebar.number_input(
        "🏷️ Discount %",
        min_value=0.0,
        max_value=100.0,
        value=float(latest_input_row["Discount_Percentage"]),
        step=1.0
    )

    promotion_options = dataset_df["Promotion_Flag"].dropna().unique().tolist()
    promotion_display = [str(x) for x in promotion_options]
    promotion_default = str(latest_input_row["Promotion_Flag"])
    promotion_input_display = st.sidebar.selectbox(
        "📣 Promotion",
        promotion_display,
        index=promotion_display.index(promotion_default) if promotion_default in promotion_display else 0
    )

    stock_input = st.sidebar.number_input(
        "📦 Stock Availability",
        min_value=0.0,
        value=float(latest_input_row["Stock_Availability"]),
        step=1.0
    )

    holiday_input = st.sidebar.selectbox(
        "🎉 Holiday",
        ["Auto (based on date)", "Yes", "No"]
    )

    local_event_options = dataset_df["Local_Event_Flag"].dropna().unique().tolist()
    local_event_display = [str(x) for x in local_event_options]
    local_event_default = str(latest_input_row["Local_Event_Flag"])
    local_event_input_display = st.sidebar.selectbox(
        "📍 Local Event",
        local_event_display,
        index=local_event_display.index(local_event_default) if local_event_default in local_event_display else 0
    )

    competitor_input = st.sidebar.number_input(
        "💰 Competitor Price",
        min_value=0.0,
        value=float(latest_input_row["Competitor_Price"]),
        step=1.0
    )

    economic_input = st.sidebar.number_input(
        "📊 Economic Indicator",
        value=float(latest_input_row["Economic_Indicator"]),
        step=0.1
    )

    marketing_input = st.sidebar.number_input(
        "📣 Marketing Spend",
        min_value=0.0,
        value=float(latest_input_row["Marketing_Spend"]),
        step=100.0
    )

    with st.sidebar.expander("Additional model features"):
        weather_options = dataset_df["Weather"].dropna().astype(str).unique().tolist()
        weather_default = str(latest_input_row["Weather"])
        weather_input = st.selectbox(
            "Weather",
            weather_options,
            index=weather_options.index(weather_default) if weather_default in weather_options else 0
        )

        channel_options = dataset_df["Sales_Channel"].dropna().astype(str).unique().tolist()
        channel_default = str(latest_input_row["Sales_Channel"])
        channel_input = st.selectbox(
            "Sales Channel",
            channel_options,
            index=channel_options.index(channel_default) if channel_default in channel_options else 0
        )

        segment_options = dataset_df["Customer_Segment"].dropna().astype(str).unique().tolist()
        segment_default = str(latest_input_row["Customer_Segment"])
        segment_input = st.selectbox(
            "Customer Segment",
            segment_options,
            index=segment_options.index(segment_default) if segment_default in segment_options else 0
        )


    def convert_to_dataset_type(column, display_value):
        """Convert Yes/No-style UI values to the dataset's stored type."""
        sample = dataset_df[column].dropna()
        if len(sample) == 0:
            return display_value
        sample_value = sample.iloc[0]
        if isinstance(sample_value, (int, np.integer, float, np.floating)):
            if str(display_value).lower() == "yes":
                return 1
            if str(display_value).lower() == "no":
                return 0
        return display_value


    user_holiday_value = None
    if holiday_input == "Yes":
        user_holiday_value = 1
    elif holiday_input == "No":
        user_holiday_value = 0

    user_inputs = {
        "Price": price_input,
        "Discount_Percentage": discount_input,
        "Promotion_Flag": convert_to_dataset_type("Promotion_Flag", promotion_input_display),
        "Stock_Availability": stock_input,
        "Local_Event_Flag": convert_to_dataset_type("Local_Event_Flag", local_event_input_display),
        "Competitor_Price": competitor_input,
        "Economic_Indicator": economic_input,
        "Marketing_Spend": marketing_input,
        "Weather": weather_input,
        "Sales_Channel": channel_input,
        "Customer_Segment": segment_input
    }

    if user_holiday_value is not None:
        user_inputs["Holiday_Flag"] = user_holiday_value


# ============================================================
# INPUTS FOR STORE TOTAL FORECAST
# ============================================================

else:

    # Store Total requires only Store Location.
    # No Region, Product, Price, Discount, Promotion, etc.
    # are requested from the user. Each product in the selected
    # store uses its latest historical feature values.

    store_location_df = (
        dataset_df[["Store_ID", "Store_Location"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Store_ID")
    )

    store_total_options = [
        f"{row['Store_ID']} - {row['Store_Location']}"
        for _, row in store_location_df.iterrows()
    ]

    selected_store_display = st.sidebar.selectbox(
        "🏬 Store Location",
        store_total_options
    )

    selected_store = selected_store_display.split(" - ", 1)[0]

    # Empty dictionary means generate_store_total_forecast()
    # will use the latest historical inputs for each product.
    user_inputs = {}


# ============================================================
# FORECAST HORIZON
# ============================================================

horizon_options = {
    "1 Day": 1,
    "7 Days": 7,
    "14 Days": 14,
    "30 Days": 30
}

selected_horizon_label = st.sidebar.selectbox(
    "Forecast Horizon",
    list(horizon_options.keys())
)

selected_horizon = horizon_options[selected_horizon_label]


# ============================================================
# FORECAST START DATE
# ============================================================

if ZoneInfo is not None:
    current_date = date.today()
    try:
        from datetime import datetime
        current_date = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    except Exception:
        current_date = date.today()
else:
    current_date = date.today()

latest_dataset_date = dataset_df["Date"].max().date()

minimum_forecast_date = max(
    current_date,
    latest_dataset_date + pd.Timedelta(days=1)
)

selected_start_date = st.sidebar.date_input(
    "Forecast Start Date",
    value=minimum_forecast_date,
    min_value=latest_dataset_date + pd.Timedelta(days=1)
)


# ============================================================
# INFORMATION MESSAGE
# ============================================================

if forecast_type == "Product + Store":
    st.sidebar.info(
        "Forecast starts on the selected Forecast Start Date and "
        "continues for the selected horizon. Region, Product, Store "
        "and business features are used for the Product + Store forecast."
    )
else:
    st.sidebar.info(
        "Store Total forecast requires only Store Location, Forecast "
        "Start Date and Forecast Horizon. The total is calculated by "
        "forecasting all products available in the selected store and "
        "summing their predicted units for each day."
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.sidebar.button(

    "🔮 Predict Sales",

    use_container_width=True

)


# ============================================================
# FORECAST EXECUTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # PRODUCT + STORE FORECAST
        # ====================================================

        if forecast_type == "Product + Store":

            forecast_df = generate_forecast(

                selected_product,

                selected_store,

                selected_horizon,

                model_history_df,

                selected_start_date,

                user_inputs

            )


            # ------------------------------------------------
            # Names
            # ------------------------------------------------

            product_name = get_product_name(
                selected_product
            )

            store_name = get_store_name(
                selected_store
            )


            # ------------------------------------------------
            # Forecast totals
            # ------------------------------------------------

            total_forecast = int(

                forecast_df[
                    "Predicted_Units_Sold"
                ].sum()

            )


            average_forecast = int(

                round(
                    forecast_df[
                        "Predicted_Units_Sold"
                    ].mean()
                )

            )


            peak_forecast = int(

                forecast_df[
                    "Predicted_Units_Sold"
                ].max()

            )


            minimum_forecast = int(

                forecast_df[
                    "Predicted_Units_Sold"
                ].min()

            )


            # ------------------------------------------------
            # Historical data
            # ------------------------------------------------

            pair_history = dataset_df[

                (dataset_df["Product_ID"] == selected_product)
                &
                (dataset_df["Store_ID"] == selected_store)

            ].sort_values(
                "Date"
            )


            latest_historical_date = (

                pair_history["Date"].max()

            )


            forecast_start_date = (

                forecast_df["Date"].min()

            )


            forecast_end_date = (

                forecast_df["Date"].max()

            )


            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                "✅ Forecast generated successfully!"
            )


            # =================================================
            # FORECAST PERIOD
            # =================================================

            st.subheader(
                "📅 Forecast Period"
            )


            period_col1, period_col2, period_col3 = (
                st.columns(3)
            )


            with period_col1:

                st.write(
                    "Latest Historical Date"
                )

                st.markdown(
                    f"### {latest_historical_date.strftime('%Y-%m-%d')}"
                )


            with period_col2:

                st.write(
                    "Forecast Start"
                )

                st.markdown(
                    f"### {forecast_start_date.strftime('%Y-%m-%d')}"
                )


            with period_col3:

                st.write(
                    "Forecast End"
                )

                st.markdown(
                    f"### {forecast_end_date.strftime('%Y-%m-%d')}"
                )


            st.divider()


            # =================================================
            # FORECAST SUMMARY CARDS
            # =================================================

            st.subheader(
                "📊 Forecast Summary"
            )


            metric_col1, metric_col2, metric_col3, metric_col4 = (
                st.columns(4)
            )


            with metric_col1:

                st.metric(
                    "Forecast Horizon",
                    f"{selected_horizon} Days"
                )


            with metric_col2:

                st.metric(
                    "Total Forecast",
                    f"{total_forecast:,} Units"
                )


            with metric_col3:

                st.metric(
                    "Daily Average",
                    f"{average_forecast:,} Units"
                )


            with metric_col4:

                st.metric(
                    "Peak Forecast",
                    f"{peak_forecast:,} Units"
                )


            st.divider()


            # =================================================
            # FORECAST DETAILS
            # =================================================

            st.subheader(
                "📊 Forecast Details"
            )


            result_df = forecast_df.copy()


            result_df["Product"] = (

                selected_product
                + " - "
                + product_name

            )


            result_df["Store"] = (

                selected_store
                + " - "
                + store_name

            )


            result_df = result_df[

                [
                    "Date",
                    "Product",
                    "Store",
                    "Predicted_Units_Sold"
                ]

            ].copy()


            result_df = result_df.rename(

                columns={

                    "Date": "Forecast Date",

                    "Predicted_Units_Sold":
                        "Predicted Units Sold"

                }

            )


            result_df["Forecast Date"] = (

                pd.to_datetime(
                    result_df["Forecast Date"]
                )
                .dt.strftime("%Y-%m-%d")

            )


            result_df[
                "Predicted Units Sold"
            ] = (

                result_df[
                    "Predicted Units Sold"
                ]
                .astype(int)

            )


            st.dataframe(

                result_df,

                use_container_width=True,

                hide_index=True

            )


            # =================================================
            # DAILY SALES FORECAST CHART
            # =================================================

            st.subheader(
                "📈 Daily Sales Forecast"
            )


            chart_forecast = forecast_df[
                [
                    "Date",
                    "Predicted_Units_Sold"
                ]
            ].copy()


            chart_forecast = (

                chart_forecast
                .set_index("Date")
                .rename(
                    columns={
                        "Predicted_Units_Sold":
                            "Forecast"
                    }
                )

            )


            st.line_chart(
                chart_forecast,
                use_container_width=True
            )


            # =================================================
            # ACTUAL VS FORECAST CHART
            # =================================================

            st.subheader(
                "📈 Actual vs Forecast Sales"
            )


            actual_history = pair_history[
                [
                    "Date",
                    "Units_Sold"
                ]
            ].copy()


            actual_history = (

                actual_history
                .groupby(
                    "Date",
                    as_index=False
                )["Units_Sold"]
                .sum()

            )


            actual_history = (

                actual_history
                .tail(60)

            )


            actual_history = (

                actual_history
                .rename(
                    columns={
                        "Units_Sold":
                            "Actual Sales"
                    }
                )

            )


            forecast_chart = forecast_df[
                [
                    "Date",
                    "Predicted_Units_Sold"
                ]
            ].copy()


            forecast_chart = (

                forecast_chart
                .rename(
                    columns={
                        "Predicted_Units_Sold":
                            "Forecast"
                    }
                )

            )


            actual_history = (

                actual_history
                .set_index("Date")

            )


            forecast_chart = (

                forecast_chart
                .set_index("Date")

            )


            combined_chart = pd.concat(

                [
                    actual_history,
                    forecast_chart
                ],

                axis=0

            ).sort_index()


            st.line_chart(

                combined_chart,

                use_container_width=True

            )


            # =================================================
            # HISTORICAL INFORMATION
            # =================================================

            st.subheader(
                "📚 Historical Information Used"
            )


            st.write(

                f"Latest historical date: "
                f"**{latest_historical_date.strftime('%Y-%m-%d')}**"

            )


            st.write(

                f"Historical observations used: "
                f"**{len(pair_history)}**"

            )


            history_display = pair_history.tail(
                10
            ).copy()


            # Add display-friendly names

            history_display["Product"] = (

                history_display["Product_ID"]
                + " - "
                + history_display["Product_Name"]

            )


            history_display["Store"] = (

                history_display["Store_ID"]
                + " - "
                + history_display["Store_Location"]

            )


            # Move Product and Store columns

            history_display = history_display[

                [
                    "Date",
                    "Product",
                    "Category",
                    "Store",
                    "Units_Sold",
                    "Price",
                    "Discount_Percentage",
                    "Promotion_Flag",
                    "Stock_Availability",
                    "Day_of_Week",
                    "Month",
                    "Quarter",
                    "Holiday_Flag",
                    "Is_Weekend",
                    "Season",
                    "Weather",
                    "Local_Event_Flag",
                    "Competitor_Price",
                    "Economic_Indicator",
                    "Sales_Channel",
                    "Customer_Segment",
                    "Marketing_Spend"
                ]

            ]


            st.dataframe(

                history_display,

                use_container_width=True,

                hide_index=True

            )


        # ====================================================
        # STORE TOTAL FORECAST
        # ====================================================

        else:

            store_total_df, product_forecast_df, skipped_products = (

                generate_store_total_forecast(

                    selected_store,

                    selected_horizon,

                    model_history_df,

                    selected_start_date,

                    user_inputs

                )

            )


            store_name = get_store_name(
                selected_store
            )


            # ------------------------------------------------
            # Store latest date
            # ------------------------------------------------

            store_history = dataset_df[
                dataset_df["Store_ID"] == selected_store
            ].copy()


            latest_historical_date = (

                store_history["Date"].max()

            )


            forecast_start_date = (

                store_total_df["Date"].min()

            )


            forecast_end_date = (

                store_total_df["Date"].max()

            )


            # ------------------------------------------------
            # Summary metrics
            # ------------------------------------------------

            total_forecast = int(

                store_total_df[
                    "Predicted_Units_Sold"
                ].sum()

            )


            average_forecast = int(

                round(
                    store_total_df[
                        "Predicted_Units_Sold"
                    ].mean()
                )

            )


            peak_forecast = int(

                store_total_df[
                    "Predicted_Units_Sold"
                ].max()

            )


            minimum_forecast = int(

                store_total_df[
                    "Predicted_Units_Sold"
                ].min()

            )


            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                "✅ Store-wise forecast generated successfully!"
            )


            # =================================================
            # STORE TITLE
            # =================================================

            st.subheader(
                f"🏪 Store Forecast - "
                f"{selected_store} - {store_name}"
            )


            # =================================================
            # FORECAST PERIOD
            # =================================================

            st.subheader(
                "📅 Forecast Period"
            )


            period_col1, period_col2, period_col3 = (
                st.columns(3)
            )


            with period_col1:

                st.write(
                    "Latest Historical Date"
                )

                st.markdown(
                    f"### {latest_historical_date.strftime('%Y-%m-%d')}"
                )


            with period_col2:

                st.write(
                    "Forecast Start"
                )

                st.markdown(
                    f"### {forecast_start_date.strftime('%Y-%m-%d')}"
                )


            with period_col3:

                st.write(
                    "Forecast End"
                )

                st.markdown(
                    f"### {forecast_end_date.strftime('%Y-%m-%d')}"
                )


            st.divider()


            # =================================================
            # FORECAST SUMMARY CARDS
            # =================================================

            st.subheader(
                "📊 Forecast Summary"
            )


            metric_col1, metric_col2, metric_col3, metric_col4 = (
                st.columns(4)
            )


            with metric_col1:

                st.metric(
                    "Forecast Horizon",
                    f"{selected_horizon} Days"
                )


            with metric_col2:

                st.metric(
                    "Total Store Forecast",
                    f"{total_forecast:,} Units"
                )


            with metric_col3:

                st.metric(
                    "Daily Store Average",
                    f"{average_forecast:,} Units"
                )


            with metric_col4:

                st.metric(
                    "Peak Daily Sales",
                    f"{peak_forecast:,} Units"
                )


            st.divider()


            # =================================================
            # DAILY TOTAL FORECAST
            # =================================================

            st.subheader(
                "📅 Daily Total Forecast"
            )


            daily_total_display = store_total_df.copy()


            daily_total_display = (

                daily_total_display.rename(

                    columns={

                        "Date":
                            "Forecast Date",

                        "Predicted_Units_Sold":
                            "Total Predicted Units Sold"

                    }

                )

            )


            daily_total_display[
                "Forecast Date"
            ] = (

                pd.to_datetime(
                    daily_total_display[
                        "Forecast Date"
                    ]
                )
                .dt.strftime("%Y-%m-%d")

            )


            daily_total_display[
                "Total Predicted Units Sold"
            ] = (

                daily_total_display[
                    "Total Predicted Units Sold"
                ]
                .astype(int)

            )


            st.dataframe(

                daily_total_display,

                use_container_width=True,

                hide_index=True

            )


            # =================================================
            # DAILY TOTAL CHART
            # =================================================

            st.subheader(
                "📈 Daily Store Sales Forecast"
            )


            store_chart = (

                store_total_df[
                    [
                        "Date",
                        "Predicted_Units_Sold"
                    ]
                ]
                .set_index("Date")
                .rename(
                    columns={
                        "Predicted_Units_Sold":
                            "Forecast"
                    }
                )

            )


            st.line_chart(

                store_chart,

                use_container_width=True

            )


            # =================================================
            # PRODUCT-WISE STORE FORECAST
            # =================================================

            st.subheader(
                "📦 Product-wise Forecast for Store"
            )


            product_forecast_display = (
                product_forecast_df.copy()
            )


            product_forecast_display[
                "Product"
            ] = (

                product_forecast_display[
                    "Product_ID"
                ].apply(
                    product_display_name
                )

            )


            product_forecast_display[
                "Store"
            ] = (

                product_forecast_display[
                    "Store_ID"
                ].apply(
                    store_display_name
                )

            )


            product_forecast_display = (

                product_forecast_display
                .groupby(
                    [
                        "Date",
                        "Product",
                        "Store"
                    ],
                    as_index=False
                )[
                    "Predicted_Units_Sold"
                ]
                .sum()

            )


            product_forecast_display[
                "Predicted_Units_Sold"
            ] = (

                product_forecast_display[
                    "Predicted_Units_Sold"
                ]
                .round()
                .astype(int)

            )


            product_forecast_display = (

                product_forecast_display
                .rename(

                    columns={

                        "Date":
                            "Forecast Date",

                        "Predicted_Units_Sold":
                            "Predicted Units Sold"

                    }

                )

            )


            product_forecast_display[
                "Forecast Date"
            ] = (

                pd.to_datetime(
                    product_forecast_display[
                        "Forecast Date"
                    ]
                )
                .dt.strftime("%Y-%m-%d")

            )


            st.dataframe(

                product_forecast_display,

                use_container_width=True,

                hide_index=True

            )


            # =================================================
            # ACTUAL VS FORECAST - STORE
            # =================================================

            st.subheader(
                "📈 Actual vs Forecast Store Sales"
            )


            actual_store = (

                store_history[
                    [
                        "Date",
                        "Units_Sold"
                    ]
                ]
                .groupby(
                    "Date",
                    as_index=False
                )["Units_Sold"]
                .sum()
                .tail(60)

            )


            actual_store = (

                actual_store
                .rename(
                    columns={
                        "Units_Sold":
                            "Actual Sales"
                    }
                )
                .set_index("Date")

            )


            store_forecast_chart = (

                store_total_df[
                    [
                        "Date",
                        "Predicted_Units_Sold"
                    ]
                ]
                .rename(
                    columns={
                        "Predicted_Units_Sold":
                            "Forecast"
                    }
                )
                .set_index("Date")

            )


            combined_store_chart = pd.concat(

                [
                    actual_store,
                    store_forecast_chart
                ],

                axis=0

            ).sort_index()


            st.line_chart(

                combined_store_chart,

                use_container_width=True

            )


            # =================================================
            # SKIPPED PRODUCTS
            # =================================================

            if len(skipped_products) > 0:

                skipped_display = [

                    product_display_name(
                        p
                    )

                    for p
                    in skipped_products

                ]


                st.warning(

                    "Some products were skipped because "
                    "they did not have enough historical "
                    "observations: "
                    + ", ".join(
                        skipped_display
                    )

                )


            # =================================================
            # STORE HISTORICAL INFORMATION
            # =================================================

            st.subheader(
                "📚 Historical Information Used"
            )


            st.write(

                f"Latest historical date: "
                f"**{latest_historical_date.strftime('%Y-%m-%d')}**"

            )


            st.write(

                f"Historical observations used: "
                f"**{len(store_history)}**"

            )


            store_history_display = (
                store_history.tail(10).copy()
            )


            store_history_display[
                "Product"
            ] = (

                store_history_display[
                    "Product_ID"
                ]
                + " - "
                + store_history_display[
                    "Product_Name"
                ]

            )


            store_history_display[
                "Store"
            ] = (

                store_history_display[
                    "Store_ID"
                ]
                + " - "
                + store_history_display[
                    "Store_Location"
                ]

            )


            store_history_display = (

                store_history_display[
                    [
                        "Date",
                        "Product",
                        "Category",
                        "Store",
                        "Units_Sold",
                        "Price",
                        "Discount_Percentage",
                        "Promotion_Flag",
                        "Stock_Availability",
                        "Day_of_Week",
                        "Month",
                        "Quarter",
                        "Holiday_Flag",
                        "Is_Weekend",
                        "Season",
                        "Weather",
                        "Local_Event_Flag",
                        "Competitor_Price",
                        "Economic_Indicator",
                        "Sales_Channel",
                        "Customer_Segment",
                        "Marketing_Spend"
                    ]
                ]

            )


            st.dataframe(

                store_history_display,

                use_container_width=True,

                hide_index=True

            )


    except Exception as e:

        st.error(
            "❌ Forecast generation failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(

    "Sales Forecasting System | "
    "Improved XGBoost | "
    "Daily Forecasting | "
    "Store-wise Forecasting | "
    "Docker-ready"

)