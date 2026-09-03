

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


######################################### BASELINE CLIMATOLOGY MODEL #####################################################

def climatology(t2m, output_path):
    time_index = pd.to_datetime(t2m.valid_time.values)
    df = pd.DataFrame(t2m.values, index=time_index, columns=['t2m'])
    #df = pd.DataFrame({"valid_time": pd.to_datetime(t2m["valid_time"].values), "t2m": t2m.values})
    #df = df.set_index("valid_time")
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]    # ----------------------------
    # Time features
    # ----------------------------
    #df["year"] = df["valid_time"].dt.year
    #df["month"] = df["valid_time"].dt.month

    ## Predict the next month temperature from a monthly average

######### Apply climatology ############################

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    #train_index = train.index
    #test_index = test.index

    # Long-term trend (months since start)

    monthly_stats = train.groupby(train.index.month)["t2m"].agg(["median", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    monthly_stats.columns = ["median", "lower", "upper"]

    prediction = pd.Series(test.index.month.map(monthly_stats["median"]).values, index=test.index)
    lower = pd.Series(test.index.month.map(monthly_stats["lower"]).values, index=test.index)
    upper = pd.Series(test.index.month.map(monthly_stats["upper"]).values, index=test.index)

    error_lower = prediction - lower
    error_upper = upper - prediction

    rmse = np.sqrt(mean_squared_error(test["t2m"], prediction))
    mae = mean_absolute_error(test["t2m"], prediction)
    r2 = r2_score(test["t2m"], prediction)

    metrics_df = pd.DataFrame({"Metric": ["RMSE", "MAE", "R2"], "Value": [rmse, mae, r2], "Unit": ["°C", "°C", "-"]})
    metrics_df.to_csv(output_path + "climatology_metrics.csv", index=False)

    prediction_df = pd.DataFrame({"date": test.index, "actual": test["t2m"].values, "prediction": prediction.values, "lower": lower.values, "upper": upper.values, "error_lower": error_lower.values, "error_upper": error_upper.values})
    prediction_df.to_csv(output_path + "climatology_predictions.csv", index=False)

    last_date = df.index[-1]
    next_month = (last_date.month % 12) + 1
    next_month_prediction = monthly_stats.loc[next_month, "median"]
    next_month_lower = monthly_stats.loc[next_month, "lower"]
    next_month_upper = monthly_stats.loc[next_month, "upper"]
    next_month_error_lower = next_month_prediction - next_month_lower
    next_month_error_upper = next_month_upper - next_month_prediction

    forecast_df = pd.DataFrame({"model": ["Climatology"], "prediction": [next_month_prediction], "lower": [next_month_lower], "upper": [next_month_upper], "error_lower": [next_month_error_lower], "error_upper": [next_month_error_upper]})
    forecast_df.to_csv(output_path + "climatology_forecast.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train["t2m"], label="Training", alpha=0.5)
    plt.plot(test.index, test["t2m"], label="Observed", color="black")
    plt.plot(test.index, prediction, label="Climatology", linewidth=2)
    plt.fill_between(test.index, lower, upper, alpha=0.25)
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.title("Monthly Climatology Forecast")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path + "Climatology_prediction.png")
    plt.close()

    print("Metrics saved to climatology_metrics.csv")
    print("Predictions saved to climatology_predictions.csv")
    return next_month_prediction

#####################################################################################################
########################### GRADIENT BOOST ##############################################################

def gradient_boost(t2m, output_path):
    df = pd.DataFrame({"valid_time": pd.to_datetime(t2m["valid_time"].values), "t2m": t2m.values})

    df["year"] = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    df["trend"] = np.arange(len(df))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in range(1, 13):
        df[f"lag_{lag}"] = df["t2m"].shift(lag)

    df["target"] = df["t2m"].shift(-1)

    feature_cols = ["trend", "month_cos", "month_sin"] + [f"lag_{lag}" for lag in range(1, 13)]

    forecast_df = df.tail(1).copy()
    X_forecast = forecast_df[feature_cols]

    df = df.dropna().reset_index(drop=True)

    X = df[feature_cols]
    y = df["target"]

    test_size = int(len(df) * 0.2)

    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    model = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=6, max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_df = pd.DataFrame({"Metric": ["RMSE", "MAE", "R2"], "Value": [rmse, mae, r2], "Unit": ["°C", "°C", "-"]})
    metrics_df.to_csv(output_path + "GradientBoost_metrics.csv", index=False)

    residuals = y_train - model.predict(X_train)
    lower_residual = np.quantile(residuals, 0.025)
    upper_residual = np.quantile(residuals, 0.975)

    lower = y_pred + lower_residual
    upper = y_pred + upper_residual
    error_lower = y_pred - lower
    error_upper = upper - y_pred

    prediction_dates = pd.to_datetime(df["valid_time"].iloc[-test_size:].values)

    prediction_output = pd.DataFrame({"date": prediction_dates, "actual": y_test.values, "prediction": y_pred, "lower": lower, "upper": upper, "error_lower": error_lower, "error_upper": error_upper})
    prediction_output.to_csv(output_path + "gradient_boost_predictions.csv", index=False)

    next_month_prediction = model.predict(X_forecast)[0]
    next_month_lower = next_month_prediction + lower_residual
    next_month_upper = next_month_prediction + upper_residual
    next_month_error_lower = next_month_prediction - next_month_lower
    next_month_error_upper = next_month_upper - next_month_prediction

    print("\nGradient Boost forecast for next month")
    print(f"{next_month_prediction:.2f} °C")
    print(f"Lower: {next_month_lower:.2f} °C")
    print(f"Upper: {next_month_upper:.2f} °C")
    print(f"Error: -{next_month_error_lower:.2f} / +{next_month_error_upper:.2f} °C")

    forecast_output = pd.DataFrame({"model": ["Gradient Boosting"], "prediction": [next_month_prediction], "lower": [next_month_lower], "upper": [next_month_upper], "error_lower": [next_month_error_lower], "error_upper": [next_month_error_upper]})
    forecast_output.to_csv(output_path + "gradient_boost_forecast.csv", index=False)

    print("Metrics saved to GradientBoost_metrics.csv")
    print("Predictions saved to gradient_boost_predictions.csv")
    return next_month_prediction


##################### SARIMA #####################################################

################################################################################
def SARIMA(t2m, output_path):
    df = pd.DataFrame({"valid_time": pd.to_datetime(t2m["valid_time"].values), "t2m": t2m.values})
    df = df.set_index("valid_time")
    y = df["t2m"].copy()

    train_size = int(len(y) * 0.8)
    train = y.iloc[:train_size]
    test = y.iloc[train_size:]

    model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()

    pred = results.get_forecast(steps=len(test))
    pred_mean = pred.predicted_mean
    conf_int = pred.conf_int()

    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]
    error_lower = pred_mean - lower
    error_upper = upper - pred_mean

    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    mae = mean_absolute_error(test, pred_mean)
    r2 = r2_score(test, pred_mean)

    metrics_df = pd.DataFrame({"Metric": ["RMSE", "MAE", "R2"], "Value": [rmse, mae, r2], "Unit": ["°C", "°C", "-"]})
    metrics_df.to_csv(output_path + "SARIMA_metrics.csv", index=False)

    prediction_output = pd.DataFrame({"date": test.index, "actual": test.values, "prediction": pred_mean.values, "lower": lower.values, "upper": upper.values, "error_lower": error_lower.values, "error_upper": error_upper.values})
    prediction_output.to_csv(output_path + "SARIMA_predictions.csv", index=False)

    final_model = SARIMAX(y, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    final_results = final_model.fit()

    forecast = final_results.get_forecast(steps=1)
    next_month_prediction = forecast.predicted_mean.iloc[0]
    next_month_conf = forecast.conf_int()
    next_month_lower = next_month_conf.iloc[0, 0]
    next_month_upper = next_month_conf.iloc[0, 1]
    next_month_error_lower = next_month_prediction - next_month_lower
    next_month_error_upper = next_month_upper - next_month_prediction

    forecast_output = pd.DataFrame({"model": ["SARIMA"], "prediction": [next_month_prediction], "lower": [next_month_lower], "upper": [next_month_upper], "error_lower": [next_month_error_lower], "error_upper": [next_month_error_upper]})
    forecast_output.to_csv(output_path + "SARIMA_forecast.csv", index=False)

    print("\nSARIMA forecast for next month")
    print(f"{next_month_prediction:.2f} °C")
    print(f"Lower: {next_month_lower:.2f} °C")
    print(f"Upper: {next_month_upper:.2f} °C")
    print(f"Error: -{next_month_error_lower:.2f} / +{next_month_error_upper:.2f} °C")

    figure_name = output_path + "SARIMA_prediction.png"
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train, label="Training")
    plt.plot(test.index, test, label="Observed", color="black")
    plt.plot(test.index, pred_mean, label="SARIMA prediction")
    plt.fill_between(test.index, lower, upper, alpha=0.25)
    plt.legend()
    plt.grid(True)
    plt.ylabel("Temperature (°C)")
    plt.title("SARIMA One-Step Forecast")
    plt.savefig(figure_name)
    plt.close()

    print("Metrics saved to SARIMA_metrics.csv")
    print("Predictions saved to SARIMA_predictions.csv")
    return next_month_prediction























