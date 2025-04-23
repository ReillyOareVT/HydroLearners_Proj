# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:02:04 2025

@author: malif
"""
# -*- coding: utf-8 -*-
"""
Streamflow and Meteorological Data Processing and Visualization

Steps:
1. Load and filter streamflow data by date and completeness
2. Identify basins with full streamflow and meteorological data
3. Resample daily meteorology to weekly
4. Convert streamflow to long format
5. Merge streamflow with meteorology
6. Analyze yearly data completeness
7. Visualize time trends and heatmaps
8. Generate % missing data matrix per variable (basin x year)
"""

#%% Import required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%% User settings
meteo_folder = r"D:\\VT_SPR2025\\CEE5984_Machine Learning in Water Resources\\GroupProject\\01_EStreams_Data\\EStreams\\meteorology"
streamflow_path = r"D:\\VT_SPR2025\\CEE5984_Machine Learning in Water Resources\\GroupProject\\01_EStreams_Data\\EStreams\\streamflow_indices\\weekly\\weekly_streamflow_mean.csv"
start_date = "1950-01-01"
end_date = "2020-12-31"

#%% Load and filter streamflow data by date
weekly_streamflow = pd.read_csv(streamflow_path, parse_dates=True, index_col=0)
streamflow_filtered = weekly_streamflow.loc[start_date:end_date]

#%% Select basins with 100% complete streamflow data
full_streamflow_basins = streamflow_filtered.columns[streamflow_filtered.notna().mean() == 1].tolist()
print(f"\nBasins with full streamflow data ({start_date} to {end_date}): {len(full_streamflow_basins)}")

#%% Define weekly resampling function for meteorology data
def resample_meteorology_weekly(df):
    df.index = pd.to_datetime(df.index)
    rules = {
        'p_mean': 'sum', 'pet_mean': 'sum',
        't_mean': 'mean', 't_min': 'mean', 't_max': 'mean',
        'sp_mean': 'mean', 'rh_mean': 'mean', 'ws_mean': 'mean', 'swr_mean': 'mean'
    }
    return df.resample('W-SUN').agg(rules)

#%% Select basins meteorology data in the same period
meteo_basins = []

for basin_id in full_streamflow_basins:
    meteo_path = os.path.join(meteo_folder, f"estreams_meteorology_{basin_id}.csv")
    if not os.path.exists(meteo_path):
        continue
    df_meteo = pd.read_csv(meteo_path, parse_dates=True, index_col=0)
    df_weekly = resample_meteorology_weekly(df_meteo.loc[start_date:end_date])
    # if df_weekly.loc[start_date:end_date].isna().any().any():
    #     continue
    meteo_basins.append(basin_id)


#%% Convert streamflow to long format
streamflow_long = streamflow_filtered[meteo_basins].reset_index().melt(
    id_vars=['index'], var_name='basin_id', value_name='streamflow'
).rename(columns={'index': 'date'})

#%% Convert meteorology to long format and merge
meteo_long_all = []
for basin_id in meteo_basins:
    path = os.path.join(meteo_folder, f"estreams_meteorology_{basin_id}.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0).loc[start_date:end_date]
    df = resample_meteorology_weekly(df).loc[start_date:end_date]
    df['basin_id'] = basin_id
    meteo_long_all.append(df.reset_index())

meteo_long = pd.concat(meteo_long_all, ignore_index=True)

#%% Merge streamflow and meteorology
estreams_merged_01 = pd.merge(streamflow_long, meteo_long, on=['date', 'basin_id'], how='inner')

#%% Add year column for yearly completeness analysis
estreams_merged_01['year'] = pd.to_datetime(estreams_merged_01['date']).dt.year

#%% Calculate % availability and missingness per year per variable
variables = ['p_mean', 't_mean', 't_min', 't_max', 'sp_mean', 'rh_mean', 'ws_mean', 'swr_mean', 'pet_mean']
years = sorted(estreams_merged_01['year'].unique())

yearly_available = {var: [] for var in variables}
yearly_missing = {var: [] for var in variables}

for year in years:
    df_year = estreams_merged_01[estreams_merged_01['year'] == year]
    for var in variables:
        available_pct = df_year[var].notna().mean() * 100
        yearly_available[var].append(available_pct)
        yearly_missing[var].append(100 - available_pct)

#%% Plot yearly % missing per variable
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

#%% Heatmap of % availability per variable over time
heatmap_df = pd.DataFrame(yearly_available, index=years).T
plt.figure(figsize=(24, 10))
sns.heatmap(
    heatmap_df, cmap="YlGnBu", annot=True, fmt=".0f",
    annot_kws={"size": 8}, linewidths=0.5, linecolor='gray',
    vmin=0, vmax=100, cbar_kws={'label': 'Data Availability (%)'}
)
plt.xlabel('Year')
plt.ylabel('Meteorological Variable')
plt.title('Data Availability Over Time (Heatmap)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Create % missing dataframes (basin x year) for each variable
basins = estreams_merged_01['basin_id'].unique()
missing_matrix = {}

for var in variables:
    df_missing = pd.DataFrame(index=basins, columns=years)
    for year in years:
        year_data = estreams_merged_01[estreams_merged_01['year'] == year]
        grouped = year_data.groupby('basin_id')[var].apply(lambda x: x.isna().mean() * 100)
        df_missing[year] = grouped
    missing_matrix[var] = df_missing
    # df_missing.to_csv(f"missing_percentage_{var}.csv")

#%% Load and combine static attribute datasets
from functools import reduce

# Path to static attribute files
attributes_folder = r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\01_EStreams_Data\EStreams\attributes\static_attributes"

# List of static files to merge
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

# Read and store each static dataframe
static_dfs = []
for file in static_files:
    path = os.path.join(attributes_folder, file)
    if os.path.exists(path):
        df_static = pd.read_csv(path)
        static_dfs.append(df_static)

# Merge all static attributes on 'basin_id'
combined_static_attributes = reduce(
    lambda left, right: pd.merge(left, right, on='basin_id', how='outer'),
    static_dfs
)

# Preview
# print("\nCombined static attributes:")
# print(combined_static_attributes.head())
#%%
static_attr_list = combined_static_attributes.columns.tolist()
#%% Select and merge specific static attributes with estreams_merged_01

# Desired attribute columns to retain
selected_static_columns = [
    'basin_id',             # join key
    'ele_mt_mean',          # Mean elevation
    'slp_dg_mean',          # Mean slope
    # 'catchment_area',       # Area
    'p_mean',               # Long-term average precipitation
    'pet_mean',             # Long-term average PET
    'aridity',              # Aridity index
    'p_seasonality',        # Seasonality
    'frac_snow',            # Snowfall fraction
    'soil_tawc',            # Soil water capacity
    'soil_bd',              # Bulk density
    'soil_fra_sand',        # Soil sand fraction
    'soil_fra_clay',        # Soil clay fraction
    'lit_dom',              # Lithology class
    'lulc_dom',             # Land use class
    'ndvi_mean',            # Mean NDVI
    'lai_mean'              # Mean LAI
]

# Subset the combined static attribute table
static_selected = combined_static_attributes[selected_static_columns].drop_duplicates(subset='basin_id')

# Merge with estreams_merged_01
estreams_merged_02= pd.merge(
    estreams_merged_01,
    static_selected,
    on='basin_id',
    how='left'
)

# Preview
# print("\nMerged dataset with static attributes:")
# print(estreams_merged_02.head())


