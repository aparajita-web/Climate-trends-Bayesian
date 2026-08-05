

import numpy as np
import matplotlib.pyplot as plt
import sys
import xarray
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from statsmodels.tsa.seasonal import STL

from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
sys.path.append("../src/climate_trends/")

import load as ld
import xarray as xr
import pandas as pd



def climatology(t2m,output_path):
    df = pd.DataFrame({
    "valid_time": pd.to_datetime(t2m["valid_time"].values),
    
    "t2m": t2m.values
        })
    df = df.set_index("valid_time")
  
    test_size = int(len(df)*0.2)
    train_size=int(len(df)*0.8)
    ######### Apply climatology ############################
    
    train = df[:train_size]
    test = df.iloc[train_size:]

    # --------------------------------------------------
    # Compute climatology from TRAIN ONLY
    # --------------------------------------------------

    monthly_climatology = train.groupby(train.index.month)["t2m"].mean()


    # --------------------------------------------------
    # Predict test period
    # --------------------------------------------------

    prediction = test.index.month.map(monthly_climatology)

    prediction = pd.Series(
        prediction.values,
        index=test.index
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    mse = mean_squared_error(test, prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, prediction)
    r2= r2_score(test, prediction)


    # Store metrics in a DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2],
        "Unit": ["°C", "°C", "-"]
    })
        # Last observation date
    last_date = df.index[-1]

    # Next month number (1-12)
    next_month = (last_date.month % 12) + 1

    # Climatology forecast
    next_month_prediction = monthly_climatology.loc[next_month]



    # Save to CSV
    metrics_filename=output_path+"climatology_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)

    print("Metrics saved to climatology_metrics.csv")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plt.figure(figsize=(12,5))

    plt.plot(
        train.index,
        train,
        label="Training",
        alpha=0.5
    )

    plt.plot(
        test.index,
        test,
        label="Observed",
        color="black"
    )

    plt.plot(
        prediction.index,
        prediction,
        label="Climatology",
        linewidth=2
    )

    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.title("Monthly Climatology Forecast")
    plt.grid(True)
    plt.legend()

    plt.savefig(output_path+"Climatology_prediction.png")
    return next_month_prediction


def gradient_boost(t2m,output_path):
   ## Process the data
    df = pd.DataFrame({
        "valid_time": pd.to_datetime(t2m["valid_time"].values),
    
        "t2m": t2m.values
            })

    ### Feature Engineering
    # ----------------------------
    # Time features
    # ----------------------------
    df["year"] = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month

    # Long-term trend (months since start)
    df["trend"] = np.arange(len(df))

    # Seasonal encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ----------------------------
    # Lag Features (Previous 12 months)
    # ----------------------------
    for lag in range(1, 13):
        df[f"lag_{lag}"] = df["t2m"].shift(lag)


    # ------------------------------------------------------------
    # Rolling Statistics
    # ------------------------------------------------------------

    df["rolling_mean_3"] = df["t2m"].shift(1).rolling(3).mean()
    df["rolling_mean_6"] = df["t2m"].shift(1).rolling(6).mean()
    df["rolling_mean_12"] = df["t2m"].shift(1).rolling(12).mean()

    df["rolling_std_3"] = df["t2m"].shift(1).rolling(3).std()
    df["rolling_std_6"] = df["t2m"].shift(1).rolling(6).std()
    df["rolling_std_12"] = df["t2m"].shift(1).rolling(12).std()



    # ----------------------------
    # Feature columns
    # ----------------------------
    #feature_cols = ["trend", "month_cos","month_sin", "rolling_mean_12","rolling_std_12", 
    #                "rolling_mean_3","rolling_std_3","rolling_mean_6","rolling_std_6",
    #    ] + [f"lag_{lag}" for lag in range(1, 13)]

    feature_cols = ["trend", "month_cos","month_sin"
        ] + [f"lag_{lag}" for lag in range(1, 13)]
    # Save the last column for forecasting
    # Row used for forecasting
    forecast_df = df.tail(1).copy()
    X_forecast=forecast_df[feature_cols]
    # Define the target variable for prediction
    df["target"] = df["t2m"].shift(-1)

    # ----------------------------
    # Remove rows with NAN values
    # ----------------------------
    df = df.dropna().reset_index(drop=True)

    ## Define the X and Y of dataset for prediction
    X = df[feature_cols]
    y = df["target"]
    ### Test train split #####
    test_size = int(len(df)*0.2)
    train_size=int(len(df)*0.8)
   
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]

    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]
    
    ## Implement Gradient Boost Model
    model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=500,
    random_state=42,
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    y_pred=model.predict(X_test)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store metrics in a DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2],
        "Unit": ["°C", "°C", "-"]
    })
    next_month_prediction = model.predict(X_forecast)[0]

    #print("\nForecast for next month")
    #print(f"{next_month_prediction:.2f} °C")
    # Save to CSV
    metrics_filename=output_path+"GradientBoost_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    print("Metrics saved to GradientBoost_metrics.csv")

    return next_month_prediction


def SARIMA(t2m,output_path):
    ## Load the dataset for Paris

    df = pd.DataFrame({
        "valid_time": pd.to_datetime(t2m["valid_time"].values),
        "t2m": t2m.values
    })
    df = df.set_index("valid_time")
    ## Convert to panda format

    y = df["t2m"].copy()


    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    train_size = int(len(y) * 0.8)

    train = y.iloc[:train_size]
    test = y.iloc[train_size:]

    model = SARIMAX(
    train,
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

    results = model.fit()

    #print(results.summary())
    # --------------------------------------------------------
    # Predict test period
    # --------------------------------------------------------

    pred = results.get_forecast(steps=len(test))

    pred_mean = pred.predicted_mean

    conf_int = pred.conf_int()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mse = mean_squared_error(test, pred_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, pred_mean)
    r2 = r2_score(test, pred_mean)
    # Store metrics in a DataFrame
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R2"],
        "Value": [rmse, mae, r2],
        "Unit": ["°C", "°C", "-"]
    })

    # --------------------------------------------------------
    # Forecast next month
    # --------------------------------------------------------
    # --------------------------------------------------------


    final_model = SARIMAX(
        y,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    final_results = final_model.fit()

    forecast = final_results.get_forecast(steps=1)

    next_month_prediction = forecast.predicted_mean.iloc[0]

    # Save to CSV
    metrics_filename=output_path+"SARIMA_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)
   
    print("Metrics saved to SARIMA_metrics.csv")

    


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    figure_name=output_path+"SARIMA_prediction.png"
    plt.figure(figsize=(12,5))

    plt.plot(train.index, train, label="Training")
    plt.plot(test.index, test, label="Observed", color="black")
    plt.plot(test.index, pred_mean, label="SARIMA prediction")

    plt.fill_between(
        test.index,
        conf_int.iloc[:,0],
        conf_int.iloc[:,1],
        alpha=0.25
    )

    plt.legend()
    plt.grid(True)
    plt.ylabel("Temperature (°C)")
    plt.title("SARIMA One-Step Forecast")
    plt.savefig(figure_name)
    return next_month_prediction



            