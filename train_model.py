import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("ev_charging_dataset.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# =====================================
# DATA CLEANING
# =====================================

df = df.drop_duplicates()

for col in df.columns:

    if df[col].dtype == "object":

        df[col].fillna(
            df[col].mode()[0],
            inplace=True
        )

    else:

        df[col].fillna(
            df[col].median(),
            inplace=True
        )


# =====================================
# EDA
# =====================================

print("\nStatistical Summary")
print(df.describe())

numeric_cols = df.select_dtypes(
    include=np.number
).columns

# Correlation Heatmap

plt.figure(figsize=(12,8))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()


# =====================================
# ROOT CAUSE ANALYSIS
# =====================================

target_column = "Charging_Load_kW"

correlation = (
    df[numeric_cols]
    .corr()[target_column]
    .sort_values(ascending=False)
)

print("\nFeatures affecting utilization:")
print(correlation)

correlation.drop(
    target_column
).plot(
    kind="bar",
    figsize=(10,5)
)

plt.title(
    "Root Cause Analysis"
)

plt.ylabel(
    "Correlation"
)

plt.show()


# =====================================
# ENCODING
# =====================================

label_encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col]
        )

        label_encoders[col] = le


# =====================================
# FEATURE SELECTION
# =====================================

target = "Charging_Load_kW"

X = df.drop(
    columns=[target]
)

y = df[target]


# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================
# MODEL COMPARISON
# =====================================

models = {

    "Linear Regression":
        LinearRegression(),

    "KNN":
        KNeighborsRegressor(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            random_state=42
        )
}

results = []

print("\nMODEL COMPARISON\n")

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append([
        name,
        mse,
        rmse,
        mae,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MSE",
        "RMSE",
        "MAE",
        "R2"
    ]
)

print(results_df)

# Model Comparison Chart

results_df.plot(
    x="Model",
    y="R2",
    kind="bar",
    figsize=(8,5)
)

plt.title(
    "Model Performance Comparison"
)

plt.ylabel(
    "R² Score"
)

plt.show()


# =====================================
# HYPERPARAMETER TUNING
# =====================================

print("\nTuning Random Forest...\n")

rf_params = {

    "n_estimators":
        [100, 200, 300],

    "max_depth":
        [10, 20, None],

    "min_samples_split":
        [2, 5, 10],

    "min_samples_leaf":
        [1, 2, 4]
}

rf_grid = GridSearchCV(

    RandomForestRegressor(
        random_state=42
    ),

    rf_params,

    cv=5,

    scoring="r2",

    n_jobs=-1
)

rf_grid.fit(
    X_train,
    y_train
)

best_rf = rf_grid.best_estimator_

print(
    "Best RF Params:",
    rf_grid.best_params_
)


# =====================================
# XGBOOST TUNING
# =====================================

print("\nTuning XGBoost...\n")

xgb_params = {

    "n_estimators":
        [100, 200],

    "max_depth":
        [3, 5, 7],

    "learning_rate":
        [0.01, 0.05, 0.1]
}

xgb_grid = GridSearchCV(

    XGBRegressor(
        random_state=42
    ),

    xgb_params,

    cv=5,

    scoring="r2",

    n_jobs=-1
)

xgb_grid.fit(
    X_train,
    y_train
)

best_xgb = xgb_grid.best_estimator_

print(
    "Best XGB Params:",
    xgb_grid.best_params_
)


# =====================================
# FINAL MODEL SELECTION
# =====================================

best_models = {

    "Random Forest":
        best_rf,

    "XGBoost":
        best_xgb
}

best_score = -999

best_model = None

for name, model in best_models.items():

    pred = model.predict(
        X_test
    )

    score = r2_score(
        y_test,
        pred
    )

    print(
        f"{name} R2:",
        score
    )

    if score > best_score:

        best_score = score

        best_model = model


print(
    "\nBest Model Selected:"
)

print(best_model)

print(
    "Best R2:",
    best_score
)


# =====================================
# FEATURE IMPORTANCE
# =====================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance = pd.DataFrame({

        "Feature":
            X.columns,

        "Importance":
            best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance")
    print(importance)

    importance.plot(
        x="Feature",
        y="Importance",
        kind="bar",
        figsize=(10,5)
    )

    plt.title(
        "Feature Importance"
    )

    plt.show()


# =====================================
# SAVE MODEL
# =====================================

joblib.dump(
    best_model,
    "model.pkl"
)

print(
    "\nmodel.pkl saved successfully!"
)
