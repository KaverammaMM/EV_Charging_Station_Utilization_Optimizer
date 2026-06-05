# EV Charging Station Utilization Optimizer

## Overview

This project uses Machine Learning to optimize EV Charging Station utilization by analyzing charging patterns, predicting charging load, estimating waiting times, identifying peak demand periods, visualizing station locations, and estimating carbon savings.

---

## Features

### Occupancy Prediction
Predict charging station utilization using Machine Learning algorithms.

### Wait Time Prediction
Estimate queue and waiting time for EV users.

### Peak Hour Analysis
Identify busiest charging hours and charging demand patterns.

### Interactive Map
Visualize EV charging stations using geospatial mapping.

### Carbon Savings Dashboard
Estimate carbon savings based on energy consumption.

---

## Dataset Features

- Charging_Rate_kW
- Queue_Time_mins
- Station_Capacity_EV
- Time_Spent_Charging_mins
- Energy_Drawn_kWh
- Session_Start_Hour
- Fleet_Size
- Temperature_C
- Traffic_Data
- Current_Latitude
- Current_Longitude

---

## Machine Learning Models

The following algorithms are used:

1. Linear Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Regressor
4. Random Forest Regressor
5. XGBoost Regressor

---

## Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## Hyperparameter Tuning

GridSearchCV is used to optimize:

- Random Forest Regressor
- XGBoost Regressor

The best model is automatically selected and saved as:

model.pkl

---

## Project Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Root Cause Analysis
4. Feature Engineering
5. Model Training
6. Hyperparameter Tuning
7. Model Comparison
8. Best Model Selection
9. Streamlit Dashboard Development
10. Deployment on Streamlit Cloud

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly
- Folium
- Streamlit

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Streamlit Dashboard Modules

- Dashboard
- Occupancy Prediction
- Wait Time Prediction
- Peak Hour Analysis
- Interactive Map
- Carbon Savings Dashboard
- Model Performance Comparison

---

## Future Enhancements

- Real-time charging station monitoring
- Dynamic pricing optimization
- Charging slot recommendation
- Demand forecasting
- Live API integration

---

## Author

EV Charging Station Utilization Optimizer

Mini Project using Machine Learning, Data Analytics, and Streamlit.
