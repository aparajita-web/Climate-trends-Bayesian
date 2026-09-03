
import sys
import yaml
import os
from statsmodels.tsa.seasonal import STL
import pandas as pd
import argparse
import mlflow 
sys.path.append("src/climate_trends/")


from download import download_era5,update_era5
import load as ld
import EDA_plot as eda
import Model_A as modA
import Model_C_ML as modC

## Load yaml configuration file

import argparse
import os
import yaml
import pandas as pd
import mlflow

# Your existing imports
# from src.climate_trends.download import update_era5
# from src.climate_trends import load_data as ld
# from src.climate_trends import eda
# from src.climate_trends import modelA as modA
# from src.climate_trends import modelC as modC
# from statsmodels.tsa.seasonal import STL


# ==========================================================
# Load YAML configuration file
# ==========================================================

parser = argparse.ArgumentParser(
    description="Run climate trend analysis"
)

parser.add_argument(
    "config",
    help="Configuration file name without .yaml"
)

args = parser.parse_args()

config_path = f"config/{args.config}.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


# ==========================================================
# Main pipeline
# ==========================================================

def main():

    # ------------------------------------------------------
    # Configuration
    # ------------------------------------------------------

    city = config["city"]

    variable = config["variable"]

    years = range(
        config["start_year"],
        config["end_year"]
    )

    area = config["area"]

    output_path = config["output_path"]

    north, west, south, east = area

    latitude = (north + south) / 2
    longitude = (west + east) / 2


    # ======================================================
    # Start MLflow experiment
    # ======================================================

    mlflow.set_experiment("Climate_Trends")

    with mlflow.start_run(
        run_name=f"{city}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        # --------------------------------------------------
        # Log configuration
        # --------------------------------------------------

        mlflow.log_param("config", args.config)
        mlflow.log_param("city", city)
        mlflow.log_param("variable", variable)

        mlflow.log_param(
            "start_year",
            config["start_year"]
        )

        mlflow.log_param(
            "end_year",
            config["end_year"]
        )

        mlflow.log_param("latitude", latitude)
        mlflow.log_param("longitude", longitude)
        mlflow.log_param("area", str(area))

        mlflow.set_tag(
            "project",
            "Climate Trends Bayesian"
        )

        mlflow.set_tag(
            "data_source",
            "ERA5"
        )

        mlflow.set_tag(
            "trend_model",
            "Bayesian"
        )

        mlflow.set_tag(
            "forecast_models",
            "Climatology, Gradient Boosting, SARIMA"
        )


        # ==================================================
        # Update ERA5 data
        # ==================================================

        print("---- Checking ERA5 data ----")

        update_era5(
            variable=variable,
            area=area,
            output_path=output_path
        )


        # ==================================================
        # Directory for storing outputs
        # ==================================================

        dir_name = f"output/{city}/"

        os.makedirs(
            dir_name,
            exist_ok=True
        )


        # ==================================================
        # Load data
        # ==================================================

        print("---- Loading ERA5 data ----")

        ds_p = ld.load_single_file(output_path)

        t2m_p = (
            ds_p["t2m"]
            .sel(
                latitude=latitude,
                longitude=longitude,
                method="nearest"
            )
            - 273.15
        )


        # ==================================================
        # Log start and end date of data
        # ==================================================

        data_start = pd.Timestamp(
            ds_p["valid_time"].min().values
        )

        data_end = pd.Timestamp(
            ds_p["valid_time"].max().values
        )

        mlflow.log_param(
            "data_start",
            str(data_start.date())
        )

        mlflow.log_param(
            "data_end",
            str(data_end.date())
        )


        # ==================================================
        # Determine next forecast month
        # ==================================================

        forecast_month = (
            data_end.to_period("M") + 1
        )

        mlflow.log_param(
            "forecast_month",
            str(forecast_month)
        )
        ##### Save the data in csv files ###########
        dataset_df = pd.DataFrame({"date": pd.to_datetime(t2m_p["valid_time"].values), "actual": t2m_p.values})
        dataset_df["year"] = dataset_df["date"].dt.year
        dataset_df["month"] = dataset_df["date"].dt.month
        dataset_df.to_csv(dir_name + "dataset.csv", index=False)

        
        # ==================================================
        # EDA analysis
        # ==================================================

        print("---- Starting EDA ----")


        # --------------------------------------------------
        # Raw monthly plot
        # --------------------------------------------------

        name_raw_monthly_plot = (
            f"{dir_name}raw_monthly.png"
        )

        eda.plot_raw_monthly(
            t2m_p,
            name_raw_monthly_plot
        )

        name_raw_monthly_plot_int = (
            f"{dir_name}raw_monthly.html"
        )

        eda.plot_raw_monthly_int(
            t2m_p,
            name_raw_monthly_plot_int
        )


        # --------------------------------------------------
        # Yearly trend
        # --------------------------------------------------

        name_raw_yearly_plot = (
            f"{dir_name}yearly_average.png"
        )

        eda.plot_yearly_trend(
            t2m_p,
            name_raw_yearly_plot
        )

        name_raw_yearly_plot_int = (
            f"{dir_name}yearly_average.html"
        )

        eda.plot_yearly_trend_int(
            t2m_p,
            name_raw_yearly_plot_int
        )


        # --------------------------------------------------
        # Anomaly trend
        # --------------------------------------------------

        name_anomaly_plot = (
            f"{dir_name}anomaly_yearly.png"
        )

        eda.plot_anomaly_trend(
            t2m_p,
            name_anomaly_plot
        )

        name_anomaly_plot_int = (
            f"{dir_name}anomaly_yearly.html"
        )

        eda.plot_anomaly_trend_int(
            t2m_p,
            name_anomaly_plot_int
        )


        # --------------------------------------------------
        # STL decomposition
        # --------------------------------------------------

        name_stl_plot = (
            f"{dir_name}stl_decomposition.png"
        )

        eda.plot_stl_decomposition(
            t2m_p,
            name_stl_plot
        )

        name_stl_plot_int = (
            f"{dir_name}stl_decomposition.html"
        )

        eda.plot_stl_decomposition_int(
            t2m_p,
            name_stl_plot_int
        )


        # --------------------------------------------------
        # Log EDA artifacts
        # --------------------------------------------------

        mlflow.log_artifact(
            name_raw_monthly_plot,
            artifact_path="eda"
        )

        mlflow.log_artifact(
            name_raw_yearly_plot,
            artifact_path="eda"
        )

        mlflow.log_artifact(
            name_anomaly_plot,
            artifact_path="eda"
        )

        mlflow.log_artifact(
            name_stl_plot,
            artifact_path="eda"
        )

        mlflow.log_artifact(
            name_raw_monthly_plot_int,
            artifact_path="eda_interactive"
        )

        mlflow.log_artifact(
            name_raw_yearly_plot_int,
            artifact_path="eda_interactive"
        )

        mlflow.log_artifact(
            name_anomaly_plot_int,
            artifact_path="eda_interactive"
        )

        mlflow.log_artifact(
            name_stl_plot_int,
            artifact_path="eda_interactive"
        )


        # ==================================================
        # Trend Analysis
        # ==================================================

        print("---- Starting trend analysis ----")


        # --------------------------------------------------
        # Data processing
        # --------------------------------------------------

        series = (
            t2m_p
            .to_series()
            .dropna()
        )

        series = series.sort_index()


        # --------------------------------------------------
        # STL decomposition
        # --------------------------------------------------

        stl_p = STL(
            series,
            period=12,
            robust=True
        )

        result = stl_p.fit()

        temp_trend = result.trend
        temp_seasonal = result.seasonal


        # --------------------------------------------------
        # Remove seasonal component
        # --------------------------------------------------

        removed_seasonal = (
            t2m_p - temp_seasonal
        )


        # --------------------------------------------------
        # Annual average
        # --------------------------------------------------

        t2m_trend_1year = (
            removed_seasonal
            .resample(
                {"valid_time": "1YE"}
            )
            .mean()
        )

        time_trend_1year = pd.to_datetime(
            t2m_trend_1year["valid_time"].values
        )

        t2m_trend_1year_values = (
            t2m_trend_1year.values
        )

        time_trend_1year_values = (
            time_trend_1year.year
        ).values


        # --------------------------------------------------
        # Normalize time axis
        # --------------------------------------------------

        time_mean = (
            time_trend_1year_values.mean()
        )

        time_std = (
            time_trend_1year_values.std()
        )

        time_norm = (
            time_trend_1year_values - time_mean
        ) / time_std


        # ==================================================
        # Bayesian Estimation — Average Temperature
        # ==================================================

        print(
            "---- Starting Bayesian Inference "
            "for trend Analysis --------"
        )

        run_name = "ModelA_average_temp"

        modA.run_Bayesian(
            time_norm,
            t2m_trend_1year_values,
            dir_name,
            run_name
        )


        posterior_df = pd.read_csv(
            dir_name
            + run_name
            + "posterior_values.csv"
        )

        beta1_row = (
            posterior_df[
                posterior_df["Parameter"] == "beta1"
            ]
            .iloc[0]
        )

        avg_trend = (
            beta1_row["Median"]
            * 10
            / time_std
        )

        avg_trend_minus = (
            beta1_row["Minus_Error"]
            * 10
            / time_std
        )

        avg_trend_plus = (
            beta1_row["Plus_Error"]
            * 10
            / time_std
        )


        # --------------------------------------------------
        # Log Bayesian average temperature metrics
        # --------------------------------------------------

        mlflow.log_metric(
            "average_warming_trend_C_per_decade",
            float(avg_trend)
        )

        mlflow.log_metric(
            "average_trend_minus_error",
            float(avg_trend_minus)
        )

        mlflow.log_metric(
            "average_trend_plus_error",
            float(avg_trend_plus)
        )


        # ==================================================
        # Bayesian Estimation — Annual Maximum Temperature
        # ==================================================

        print(
            "---- Bayesian analysis of annual "
            "maximum temperature ----"
        )

        t2m_annual_max = (
            t2m_p
            .groupby("valid_time.year")
            .max(dim="valid_time")
        )

        anual_max = t2m_annual_max.values

        run_name_max = "ModelA_max_temp"

        modA.run_Bayesian(
            time_norm,
            t2m_annual_max,
            dir_name,
            run_name_max
        )


        posterior_df = pd.read_csv(
            dir_name
            + run_name_max
            + "posterior_values.csv"
        )

        beta1_row = (
            posterior_df[
                posterior_df["Parameter"] == "beta1"
            ]
            .iloc[0]
        )

        ext_trend = (
            beta1_row["Median"]
            * 10
            / time_std
        )

        ext_trend_minus = (
            beta1_row["Minus_Error"]
            * 10
            / time_std
        )

        ext_trend_plus = (
            beta1_row["Plus_Error"]
            * 10
            / time_std
        )


        # --------------------------------------------------
        # Log extreme temperature metrics
        # --------------------------------------------------

        mlflow.log_metric(
            "extreme_warming_trend_C_per_decade",
            float(ext_trend)
        )

        mlflow.log_metric(
            "extreme_trend_minus_error",
            float(ext_trend_minus)
        )

        mlflow.log_metric(
            "extreme_trend_plus_error",
            float(ext_trend_plus)
        )


        # ==================================================
        # Save trend results
        # ==================================================

        trend_df = pd.DataFrame({

            "Average_Trend": [avg_trend],

            "Average_Minus_Error": [
                avg_trend_minus
            ],

            "Average_Plus_Error": [
                avg_trend_plus
            ],

            "Extreme_Trend": [ext_trend],

            "Extreme_Minus_Error": [
                ext_trend_minus
            ],

            "Extreme_Plus_Error": [
                ext_trend_plus
            ]

        })

        trend_path = (
            dir_name
            + "warming_trend_per_decade.csv"
        )

        trend_df.to_csv(
            trend_path,
            index=False
        )


        # --------------------------------------------------
        # Log trend results
        # --------------------------------------------------

        mlflow.log_artifact(
            trend_path,
            artifact_path="trend_analysis"
        )

        mlflow.log_artifact(
            dir_name
            + run_name
            + "posterior_values.csv",
            artifact_path="bayesian"
        )

        mlflow.log_artifact(
            dir_name
            + run_name_max
            + "posterior_values.csv",
            artifact_path="bayesian"
        )

        
        # ==================================================
        # Forecasting
        # ==================================================

        print(
            "---- Predicting next month temperature ----"
        )


        # --------------------------------------------------
        # Climatology
        # --------------------------------------------------

        pred_climatology = (
            modC.climatology(
                t2m_p,
                dir_name
            )
        )


        # --------------------------------------------------
        # Gradient Boosting
        # --------------------------------------------------

        pred_gb = (
            modC.gradient_boost(
                t2m_p,
                dir_name
            )
        )


        # --------------------------------------------------
        # SARIMA
        # --------------------------------------------------

        pred_sarima = (
            modC.SARIMA(
                t2m_p,
                dir_name
            )
        )


        # --------------------------------------------------
        # Forecast dataframe
        # --------------------------------------------------

        forecast_df = pd.DataFrame({

            "Forecast_Month": [
                str(forecast_month)
            ],

            "Climatology": [
                pred_climatology
            ],

            "Gradient_Boost": [
                pred_gb
            ],

            "SARIMA": [
                pred_sarima
            ]

        })

        forecast_path = (
            dir_name
            + "forecast_next_month.csv"
        )

        forecast_df.to_csv(
            forecast_path,
            index=False
        )


        # --------------------------------------------------
        # Log forecast metrics
        # --------------------------------------------------

        mlflow.log_metric(
            "next_month_temperature_climatology_C",
            float(pred_climatology)
        )

        mlflow.log_metric(
            "next_month_temperature_gradient_boost_C",
            float(pred_gb)
        )

        mlflow.log_metric(
            "next_month_temperature_sarima_C",
            float(pred_sarima)
        )


        # --------------------------------------------------
        # Log forecast artifact
        # --------------------------------------------------

        mlflow.log_artifact(
            forecast_path,
            artifact_path="forecast"
        )


        # ==================================================
        # Final output
        # ==================================================

        print("\n---- Forecast Results ----")
        print(forecast_df)

        print("\n---- MLflow run completed ----")

        print(
            "Run ID:",
            mlflow.active_run().info.run_id
        )

        ##### Add all the data specifications in csv file 
        # Save data specification
        data_start = pd.Timestamp(ds_p["valid_time"].min().values)
        data_end = pd.Timestamp(ds_p["valid_time"].max().values)
        n_observations = len(t2m_p)
        train_size = int(n_observations * 0.8)
        train_start = data_start
        train_end = pd.Timestamp(t2m_p.valid_time.values[train_size - 1])
        test_start = pd.Timestamp(t2m_p.valid_time.values[train_size])
        test_end = data_end

        #### Check missing months
        # Check for missing months
        dates = pd.to_datetime(ds_p["valid_time"].values)
        expected_dates = pd.date_range(start=dates.min(), end=dates.max(), freq="MS")
        missing_months = expected_dates.difference(dates)
        n_missing_months = len(missing_months)

        if n_missing_months == 0:
            print("Data completeness: 100% — no missing months.")
        else:
            print(f"WARNING: {n_missing_months} month(s) missing:")
            print(missing_months.strftime("%Y-%m").tolist())
        data_completeness = f"{1 - (n_missing_months / len(expected_dates))* 100:.2f}%"


        data_info = pd.DataFrame({"Specification": ["City", "Variable", "Latitude", "Longitude", "Data Start", 
                                                    "Data End", "Training Start", "Training End", "Testing Start", "Testing End", 
                                                    "Total Observations", "Training Observations", "Testing Observations", "Frequency",
                                                    "Missing Months","Data Completeness"], 
                                                    "Value": [city, variable, latitude, longitude, 
                                                              str(data_start.date()), str(data_end.date()), str(train_start.date()), 
                                                              str(train_end.date()), str(test_start.date()), str(test_end.date()), 
                                                              n_observations, train_size, n_observations - train_size, "Monthly", n_missing_months,data_completeness 
                                                                ]})

        data_info.to_csv(dir_name + "data_info.csv", index=False)
        #print(f"Data specification saved to {dir_name}data_info.csv")


# ==========================================================
# Run pipeline
# ==========================================================

if __name__ == "__main__":
    main()