# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:07:35 2025

@author: malif
"""


# =============================================================================
# 1. Import Required Libraries
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from functools import reduce
#%% 2 Define User Settings and Paths

meteo_folder = r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\01_EStreams_Data\EStreams\meteorology"
streamflow_path = r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\01_EStreams_Data\EStreams\streamflow_indices\weekly\weekly_streamflow_mean.csv"
attributes_folder = r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\01_EStreams_Data\EStreams\attributes\static_attributes"
catchment_area_path = r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\02_GIS\estream_catchments.csv"

start_date = "1950-01-01"
end_date = "2020-12-31"
#%% 3. Load and Filter Streamflow Data

weekly_streamflow = pd.read_csv(streamflow_path, index_col=0, parse_dates=True)
streamflow_filtered = weekly_streamflow.loc[start_date:end_date]
#%% 4. Identify Qualified Basins (≥50 years of ≥95% completeness)

weekly_threshold = 1.0
min_years_required = 50

streamflow_with_year = streamflow_filtered.copy()
streamflow_with_year['year'] = streamflow_with_year.index.year

valid_counts = streamflow_with_year.groupby('year').apply(lambda df: df.drop(columns='year').notna().mean())
basin_year_counts = (valid_counts >= weekly_threshold).sum()

qualified_basins = basin_year_counts[basin_year_counts >= min_years_required].index.tolist()
print(f"\nQualified basins: {len(qualified_basins)}")

#%% 5. Load and Resample Daily Meteorology

def load_and_resample_meteorology(basin_ids):
    all_daily = []
    for basin_id in tqdm(basin_ids, desc="Loading meteorology files"):
        path = os.path.join(meteo_folder, f"estreams_meteorology_{basin_id}.csv")
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df['basin_id'] = basin_id
            all_daily.append(df.reset_index().rename(columns={'index': 'date'}))
        except Exception as e:
            print(f"Failed to load {basin_id}: {e}")
    return pd.concat(all_daily, ignore_index=True)
#%%
df_meteo_all = load_and_resample_meteorology(qualified_basins)

#%%# Resample daily meteorology to weekly
df_meteo_all['date'] = pd.to_datetime(df_meteo_all['date'])
df_meteo_all = df_meteo_all.set_index('date')

def resample_weekly(group):
    return pd.DataFrame({
        'p_mean': group['p_mean'].resample('W-SUN').sum(min_count=7),
        'pet_mean': group['pet_mean'].resample('W-SUN').sum(min_count=7),
        't_mean': group['t_mean'].resample('W-SUN').mean(),
        't_min': group['t_min'].resample('W-SUN').mean(),
        't_max': group['t_max'].resample('W-SUN').mean(),
        'sp_mean': group['sp_mean'].resample('W-SUN').mean(),
        'rh_mean': group['rh_mean'].resample('W-SUN').mean(),
        'ws_mean': group['ws_mean'].resample('W-SUN').mean(),
        'swr_mean': group['swr_mean'].resample('W-SUN').mean()
    })

weekly_meteo = df_meteo_all.groupby('basin_id').apply(resample_weekly).reset_index().rename(columns={'level_1': 'date'})
weekly_meteo = weekly_meteo[(weekly_meteo['date'] >= start_date) & (weekly_meteo['date'] <= end_date)]
#%% 6. Merge Streamflow and Meteorological Data

streamflow_long = streamflow_filtered[qualified_basins].reset_index().melt(
    id_vars=['index'], var_name='basin_id', value_name='streamflow'
).rename(columns={'index': 'date'})

estreams_weekly_merged_01 = pd.merge(
    streamflow_long,
    weekly_meteo,
    on=['date', 'basin_id'],
    how='inner'
)

#%%
# estreams_weekly_merged_01.to_feather('estreams_weekly_merged_02.feather')
# estreams_weekly_merged_01 = pd.read_feather(r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\DataAnalysis\WeeklyTSForecasting\estreams_weekly_merged_01.feather")
# estreams_cleaned_01 = pd.read_feather(r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\DataAnalysis\WeeklyTSForecasting\estreams_cleaned_01.feather")
# estreams_weekly_merged_01 = pd.read_feather('estreams_weekly_merged_250basins.feather')

#%% Filtering MET Datasets
# estreams_weekly_merged_01 = estreams_weekly_merged_01.drop('ws_mean', axis=1)
# List of meteorological variables to chec
meteo_vars = [ 'streamflow','ws_mean', 
    'p_mean', 'pet_mean', 't_mean', 't_min', 't_max',
    'sp_mean', 'rh_mean', 'swr_mean'
]

# 1. Add 'year' column
estreams_weekly_merged_01['year'] = estreams_weekly_merged_01['date'].dt.year

# 2. Compute % missing for each basin-year per variable
missing_pct = estreams_weekly_merged_01.groupby(['basin_id', 'year'])[meteo_vars].apply(lambda g: g.isna().mean())
#%%
# Identify bad basin-years where ANY variable exceeds threshold
threshold = 0.0001  # 5% missing
bad_basin_years = (
    missing_pct[missing_pct > threshold]
    .dropna(how='all')
    .reset_index()[['basin_id', 'year']]
)

# 4. Create a column to match against
estreams_weekly_merged_01['basin_year'] = estreams_weekly_merged_01['basin_id'].astype(str) + "_" + estreams_weekly_merged_01['year'].astype(str)
bad_basin_years['basin_year'] = bad_basin_years['basin_id'].astype(str) + "_" + bad_basin_years['year'].astype(str)

# 5. Remove rows from bad years
estreams_weekly_merged_01 = estreams_weekly_merged_01[~estreams_weekly_merged_01['basin_year'].isin(bad_basin_years['basin_year'])].drop(columns='basin_year')

# 6. Optional: reset index
estreams_weekly_merged_01 = estreams_weekly_merged_01.reset_index(drop=True)

#%%.
# estreams_weekly_merged_01.to_feather("estreams_merged_1705b_70y.feather")

#%% Visualize Missing Data Trends

variables = ['p_mean', 't_mean', 't_min', 't_max', 'sp_mean', 'rh_mean', 'swr_mean', 'pet_mean', 'ws_mean']
years = sorted(estreams_weekly_merged_01['date'].dt.year.unique())

yearly_missing = {var: [] for var in variables}

for year in years:
    df_year = estreams_weekly_merged_01[estreams_weekly_merged_01['date'].dt.year == year]
    for var in variables:
        yearly_missing[var].append(100 - df_year[var].notna().mean() * 100)

plt.figure(figsize=(14, 8))
for var in variables:
    plt.plot(years, yearly_missing[var], label=var)
plt.xlabel('Year')
plt.ylabel('Missing Data (%)')
plt.title('Yearly % Missing Data Across All Basins')
plt.legend(title='Variable')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Create % missing matrix (basin x year) for each variable
basins = estreams_weekly_merged_01['basin_id'].unique()
missing_matrix = {}
estreams_weekly_merged_01['year'] = pd.to_datetime(estreams_weekly_merged_01['date']).dt.year
for var in variables:
    df_missing = pd.DataFrame(index=basins, columns=years)
    for year in years:
        year_data = estreams_weekly_merged_01[estreams_weekly_merged_01['year'] == year]
        grouped = year_data.groupby('basin_id')[var].apply(lambda x: x.isna().mean() * 100)
        df_missing[year] = grouped
    missing_matrix[var] = df_missing

#%%
# Remove problematic basins and parameters
estreams_weekly_merged_01 = estreams_weekly_merged_01[~estreams_weekly_merged_01['basin_id'].isin(['GB000410', 'NO000136', 'NO000180'])]

estreams_weekly_merged_01 = estreams_weekly_merged_01.drop('ws_mean', axis=1)
#%% Merge Static Attributes

static_files = [
    'estreams_topography_attributes.csv',
    'estreams_soil_attributes.csv',
    'estreams_geology_attributes.csv',
    'estreams_hydrology_attributes.csv',
    'estreams_vegetation_attributes.csv',
    'estreams_snowcover_attributes.csv',
    'estreams_landcover_attributes.csv',
    'estreams_geologycontinental_attributes.csv'
]
#%%
static_dfs = [pd.read_csv(os.path.join(attributes_folder, file)) for file in static_files if os.path.exists(os.path.join(attributes_folder, file))]
combined_static = reduce(lambda left, right: pd.merge(left, right, on='basin_id', how='outer'), static_dfs)

selected_static_columns = [
    'basin_id', 'ele_mt_mean', 'slp_dg_mean', 'strm_dens', 'flat_area_fra', 'steep_area_fra',
    'soil_tawc_mean', 'soil_bd_mean', 'soil_fra_sand_mean', 'soil_fra_clay_mean', 'root_dep_mean',
    'bedrk_dep', 'dam_num', 'res_num', 'lakes_tot_area', 'ndvi_mean', 'lai_mean'
]

static_selected = combined_static[selected_static_columns].drop_duplicates(subset='basin_id')
estreams_weekly_merged_02 = pd.merge(estreams_weekly_merged_01, static_selected, on='basin_id', how='left')

#%% Merge Catchment Area
df_catchment_attributes = pd.read_csv(catchment_area_path)
estreams_weekly_merged_02 = pd.merge(estreams_weekly_merged_02, df_catchment_attributes[['basin_id', 'area_estre']], on='basin_id', how='left')

#%% Check NA counts per parameter
na_counts = estreams_weekly_merged_02.isna().sum().sort_values(ascending=False)

#%% Final Cleaning
estreams_cleaned_01 = estreams_weekly_merged_02.dropna()

#save for later
# estreams_cleaned_01.to_feather("estreams_dyn_sta_merged_1705b_50year.feather")


#%% Standardization and Feature Engineering

estreams_cleaned_02 = estreams_weekly_merged_02.dropna()
estreams_std_01 = estreams_cleaned_02.copy()
numeric_cols = estreams_std_01.select_dtypes(include='number').columns.tolist()
predictor_cols = [col for col in numeric_cols if col not in ['streamflow']]

#%% Standardize predictors

scaler = StandardScaler()
estreams_std_01[predictor_cols] = scaler.fit_transform(estreams_std_01[predictor_cols])

#%% Standardize streamflow per basin

streamflow_stats = estreams_std_01.groupby('basin_id')['streamflow'].agg(['mean', 'std'])
estreams_std_01 = estreams_std_01.merge(streamflow_stats.rename(columns={'mean': 'sf_mean', 'std': 'sf_std'}), on='basin_id', how='left')
estreams_std_01['streamflow_std'] = (estreams_std_01['streamflow'] - estreams_std_01['sf_mean']) / estreams_std_01['sf_std']
estreams_std_01 = estreams_std_01.drop(columns=['sf_mean', 'sf_std'])

#%% Add cyclical features

estreams_std_01['doy_sin'] = np.sin(2 * np.pi * estreams_std_01['date'].dt.dayofyear / 365.25)
estreams_std_01['doy_cos'] = np.cos(2 * np.pi * estreams_std_01['date'].dt.dayofyear / 365.25)

#%% Reckecking the predictors
print(predictor_cols)

#%%
basin_ids = estreams_std_01['basin_id'].unique()

#%% Exporting Final Dataset

estreams_std_01.to_feather("estreams_std_V02_250b_20y.feather")

#%% Function for plotting yearly data
import matplotlib.pyplot as plt
import pandas as pd

def plot_basin_year(df, basin_id, year, variables):

    # Filter the data
    mask = (df['basin_id'] == basin_id) & (df['date'].dt.year == year)
    df_sel = df.loc[mask, ['date'] + variables].sort_values('date')

    # Check if data exists
    if df_sel.empty:
        print(f"No data for basin {basin_id} in {year}")
        return

    # Plot
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=True)
    if n == 1:
        axes = [axes]  # make it iterable
    
    for i, var in enumerate(variables):
        axes[i].plot(df_sel['date'], df_sel[var], label=var, color='tab:blue')
        axes[i].set_ylabel(var)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{basin_id} — {year}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
# Example usage
plot_basin_year(
    df= df_meteo_all , 
    basin_id="AT000340", 
    year=2015, 
    variables=["p_mean"]
)
#%% Function for plotting yearly data - multiple variables
def plot_basin_year_multi_df(var_to_df, basin_id, year):
    variables = list(var_to_df.keys())
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=True)
    if n == 1:
        axes = [axes]  

    for i, var in enumerate(variables):
        df = var_to_df[var]
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        # Filter
        mask = (df['basin_id'] == basin_id) & (df['date'].dt.year == year)
        df_sel = df.loc[mask, ['date', var]].sort_values('date')
        if df_sel.empty:
            print(f"⚠️ No data for {var} in {basin_id} during {year}")
            continue
        # Plot
        axes[i].plot(df_sel['date'], df_sel[var], label=var, color='tab:blue')
        axes[i].set_ylabel(var)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"{basin_id} — {year}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
# Example usage
plot_basin_year_multi_df(
    var_to_df={
        'streamflow': estreams_weekly_merged_02 ,
        'p_mean': estreams_weekly_merged_02 ,
        'p_mean': df_meteo_all
    },
    basin_id='AT000340',
    year=2015
)
#%% Comparing the resampled data

def compare_daily_weekly_met(daily_df, weekly_df, basin_id, year, variables):

    daily_df['date'] = pd.to_datetime(daily_df['date'])
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    daily_sel = daily_df[(daily_df['basin_id'] == basin_id) & 
                         (daily_df['date'].dt.year == year)]
    weekly_sel = weekly_df[(weekly_df['basin_id'] == basin_id) & 
                           (weekly_df['date'].dt.year == year)]

    # Plot
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5*n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, var in enumerate(variables):
        if var in daily_sel.columns and var in weekly_sel.columns:
            axes[i].plot(daily_sel['date'], daily_sel[var], label='Daily', alpha=0.6, color='tab:blue')
            axes[i].plot(weekly_sel['date'], weekly_sel[var], label='Weekly (Resampled)', marker='o', linestyle='-', color='tab:orange')
            axes[i].set_ylabel(var)
            axes[i].legend()
            axes[i].grid(True)
        else:
            axes[i].text(0.5, 0.5, f"{var} not found", transform=axes[i].transAxes, ha='center')

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Comparison of Daily vs Weekly Meteorology\n{basin_id} - {year}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
# Example usage
compare_daily_weekly_met(
    daily_df = df_meteo_all.reset_index(),  # your daily met data
    weekly_df = estreams_weekly_merged_02 ,           # your weekly resampled met data
    basin_id = 'AT000340',
    year = 2013,
    variables = ['p_mean', 'pet_mean', 't_min']
)

#%% Streamflow plotted with met variables
def compare_daily_weekly_met_with_streamflow(daily_df, weekly_df, streamflow_df, basin_id, year, variables):
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    streamflow_df['date'] = pd.to_datetime(streamflow_df['date'])

    # Filter data by basin and year
    daily_sel = daily_df[(daily_df['basin_id'] == basin_id) & (daily_df['date'].dt.year == year)]
    weekly_sel = weekly_df[(weekly_df['basin_id'] == basin_id) & (weekly_df['date'].dt.year == year)]
    stream_sel = streamflow_df[(streamflow_df['basin_id'] == basin_id) & (streamflow_df['date'].dt.year == year)]

    # Plot
    n = len(variables) + 1  # +1 for streamflow
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5*n), sharex=True)

    if n == 1:
        axes = [axes]

    # Plot streamflow (top)
    axes[0].plot(stream_sel['date'], stream_sel['streamflow'], label='Streamflow', color='tab:blue')
    axes[0].set_ylabel("Streamflow")
    axes[0].legend()
    axes[0].grid(True)

    # Plot each meteorological variable
    for i, var in enumerate(variables):
        idx = i + 1  # because 0 is streamflow
        if var in daily_sel.columns and var in weekly_sel.columns:
            axes[idx].plot(daily_sel['date'], daily_sel[var], label='Daily', alpha=0.6, color='tab:blue')
            axes[idx].plot(weekly_sel['date'], weekly_sel[var], label='Weekly (Resampled)', marker='o', linestyle='-', color='tab:orange')
            axes[idx].set_ylabel(var)
            axes[idx].legend()
            axes[idx].grid(True)
        else:
            axes[idx].text(0.5, 0.5, f"{var} not found", transform=axes[idx].transAxes, ha='center')

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Daily vs Weekly Meteorology + Streamflow\n{basin_id} - {year}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
# Example usage
compare_daily_weekly_met_with_streamflow(
    daily_df = df_meteo_all.reset_index(),        # daily meteorology
    weekly_df = estreams_weekly_merged_02,                 # resampled weekly meteo
    streamflow_df = estreams_weekly_merged_02,                 # streamflow with date/basin_id/streamflow
    basin_id = 'SE000061',
    year = 2014,
    variables = ['p_mean', 't_mean']
)
#%% Checking Skewness of the static variables
import pandas as pd

static_cols = [
    'ele_mt_mean', 'slp_dg_mean', 'strm_dens', 'flat_area_fra', 'steep_area_fra',
    'soil_tawc_mean', 'soil_bd_mean', 'soil_fra_sand_mean', 'soil_fra_clay_mean',
    'root_dep_mean', 'bedrk_dep', 'dam_num', 'res_num', 'lakes_tot_area',
    'ndvi_mean', 'lai_mean', 'area_estre'
]

# Drop duplicates 
static_df = estreams_weekly_merged_02[['basin_id'] + static_cols].drop_duplicates(subset='basin_id')

# Compute skewness
skewness = static_df[static_cols].skew().sort_values(ascending=False)

# Display top skewed variables
print(skewness)

# Optional: highlight heavily skewed features
threshold = 1.0
skewed_features = skewness[abs(skewness) > threshold]
print(skewed_features)

#%% Clusering results
import pandas as pd

cluster_df = pd.read_csv('cluster_loc.csv')  # Adjust path

cluster_counts = cluster_df['KMeans Cluster'].value_counts().sort_index().rename('total_basins')
final_basins = estreams_weekly_merged_02['basin_id'].unique()
final_cluster_df = cluster_df[cluster_df['Basin ID'].isin(final_basins)]
final_cluster_counts = final_cluster_df['KMeans Cluster'].value_counts().sort_index().rename('basins_in_final')

# Combine into a summary matrix
summary_matrix = pd.concat([cluster_counts, final_cluster_counts], axis=1).fillna(0).astype(int)
summary_matrix.index.name = 'cluster_id'
summary_matrix.reset_index(inplace=True)

# Display
print(summary_matrix)


