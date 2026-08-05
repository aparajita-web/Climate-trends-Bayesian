
import sys
import yaml
import os
from statsmodels.tsa.seasonal import STL
import pandas as pd

sys.path.append("src/climate_trends/")


from download import download_era5
import load as ld
import EDA_plot as eda
import Model_A as modA
import Model_C_ML as modC

## Load yaml configuration file

with open("config/config_Hanoi.yaml", "r") as f:
    config = yaml.safe_load(f)

city=config["city"]
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


### Download and save the data #########

download_era5(
    variable=variable,
    years=years,
    area= area,
    output_path=output_path,
    overwrite=True
)

###### Directory for storing outputs #####
dir_name="output/%s/"%city
os.makedirs(dir_name,exist_ok=True)

###### EDA analysis #####
ds_p=ld.load_single_file(output_path)
t2m_p = ds_p["t2m"].sel(latitude=latitude,longitude=longitude,method='nearest') -273.15

name_raw_monthly_plot="output/%s/raw_monthly.png"%city
eda.plot_raw_monthly(t2m_p,name_raw_monthly_plot)
name_raw_monthly_plot_int="output/%s/raw_monthly.html"%city
eda.plot_raw_monthly_int(t2m_p,name_raw_monthly_plot_int)
####################
name_raw_yearly_plot="output/%s/yearly_average.png"%city
eda.plot_yearly_trend(t2m_p,name_raw_yearly_plot)
name_raw_yearly_plot_int="output/%s/yearly_average.html"%city
eda.plot_yearly_trend_int(t2m_p,name_raw_yearly_plot_int)
##############################
name_anomaly_plot="output/%s/anomaly_yearly.png"%city
eda.plot_anomaly_trend(t2m_p,name_anomaly_plot)
name_anomaly_plot_int="output/%s/anomaly_yearly.html"%city
eda.plot_anomaly_trend_int(t2m_p,name_anomaly_plot_int)
##########################################
name_stl_plot="output/%s/stl_decomposition.png"%city
eda.plot_stl_decomposition(t2m_p,name_stl_plot)
name_stl_plot_int="output/%s/stl_decomposition.html"%city
eda.plot_stl_decomposition_int(t2m_p,name_stl_plot_int)
####### Trend Analysis #########################
#### Part I: Data Processing ###################
series = t2m_p.to_series().dropna()
series = series.sort_index()
#####
stl_p = STL(series, period=12, robust=True)  # monthly data 
result = stl_p.fit()
temp_trend=result.trend
temp_seasonal=result.seasonal

##### Subtract Seasonal Trend ######
removed_seasonal=t2m_p-temp_seasonal

##### Take annual average ##########
t2m_trend_1year = removed_seasonal.resample({"valid_time": "1YE"}).mean()
time_trend_1year=pd.to_datetime(t2m_trend_1year["valid_time"].values)
t2m_trend_1year_values=t2m_trend_1year.values
time_trend_1year_values=(time_trend_1year.year).values

### Normalize time axis ##############
time_mean = time_trend_1year_values.mean()
time_std  = time_trend_1year_values.std()
time_norm = (time_trend_1year_values - time_mean) / time_std

###### PART II: Bayesian Estimation ####################
## Define the number of walkers and initial parameters
print("---- Starting Bayesian Inference for trend Analysis --------")
run_name="ModelA_average_temp"
modA.run_Bayesian(time_norm,t2m_trend_1year_values,dir_name,run_name)

posterior_df = pd.read_csv(dir_name+run_name+ "posterior_values.csv")

beta1_row = posterior_df.loc[posterior_df["Parameter"] == "beta1"].iloc[0]

avg_trend = beta1_row["Median"]*10/time_std
avg_trend_minus = beta1_row["Minus_Error"]*10/time_std
avg_trend_plus = beta1_row["Plus_Error"]*10/time_std
#####################


### Trend Analysis for Anual maximum temperature
# Annual maximum temperature
t2m_annual_max = t2m_p.groupby('valid_time.year').max(dim='valid_time')
anual_max=t2m_annual_max.values
run_name_max="ModelA_max_temp"
modA.run_Bayesian(time_norm,t2m_annual_max,dir_name,run_name_max)


posterior_df = pd.read_csv(dir_name+run_name_max+ "posterior_values.csv")

beta1_row = posterior_df.loc[posterior_df["Parameter"] == "beta1"].iloc[0]

ext_trend = beta1_row["Median"]*10/time_std
ext_trend_minus = beta1_row["Minus_Error"]*10/time_std
ext_trend_plus = beta1_row["Plus_Error"]*10/time_std

trend_df = pd.DataFrame({
    "Average_Trend": [avg_trend],
    "Average_Minus_Error": [avg_trend_minus],
    "Average_Plus_Error": [avg_trend_plus],
    "Extreme_Trend": [ext_trend],
    "Extreme_Minus_Error": [ext_trend_minus],
    "Extreme_Plus_Error": [ext_trend_plus]
})

trend_df.to_csv(
    dir_name  + "warming_trend_per_decade.csv",
    index=False
)
##############################################################
#### PART III: State Space Model trend Analyssis
 
# ###########################################################
# PART VI:  Predicting the Next Month temperature #####################

#VIa Climatology

pred_climatology=modC.climatology(t2m_p,dir_name)
pred_gb=modC.gradient_boost(t2m_p,dir_name)
pred_sarima=modC.SARIMA(t2m_p,dir_name) 

forecast_df = pd.DataFrame({
    "Climatology": [pred_climatology],
    "Gradient_Boost": [pred_gb],
    "SARIMA": [pred_sarima]
})

forecast_df.to_csv(
    dir_name+"/forecast_next_month.csv",
    index=False
)

print(forecast_df)