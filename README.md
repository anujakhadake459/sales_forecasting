# Sales Forecasting System — GitHub Pages

This project is the browser version of the supplied Streamlit sales forecasting application.

## Included
- Product + Store forecast
- Store Total forecast
- Region → Store filtering
- Product selection
- Price, Discount %, Promotion, Stock Availability
- Holiday, Local Event, Competitor Price
- Economic Indicator, Marketing Spend
- Weather, Sales Channel, Customer Segment
- Forecast start date
- 1 / 7 / 14 / 30 day horizons
- Recursive XGBoost forecasting
- Forecast summary
- Forecast details
- Daily forecast chart
- Actual vs Forecast chart
- Historical information table

## Model
The supplied `final_xgboost_model.pkl` was converted to native XGBoost JSON so the model can run in JavaScript inside a browser. The supplied feature list contains 81 features.

## GitHub Pages
Upload the complete contents of this folder to a GitHub repository and enable GitHub Pages from:
Settings → Pages → Deploy from a branch → main → / (root)

The live page will be:
`https://YOUR-USERNAME.github.io/YOUR-REPOSITORY/`
