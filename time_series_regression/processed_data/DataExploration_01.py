# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:01:59 2025

@author: malif
"""

# -*- coding: utf-8 -*-
"""
Data availability analysis for EStreams project
- Analyze streamflow and meteorology data completeness
- Identify basins with full data coverage
Created on Mon Apr 21, 2025
@author: malif
"""

#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import os

#%% Load weekly streamflow data
streamflow_file = r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\01_EStreams_Data\EStreams\streamflow_indices\weekly\weekly_streamflow_mean.csv"
streamflow_df = pd.read_csv(streamflow_file, parse_dates=True, index_col=0)

# List of basin IDs
basin_ids = streamflow_df.columns.tolist()

#%% Filter streamflow data for 1950–2020 period
streamflow_start = "1940-01-01"
streamflow_end = "2020-12-31"

streamflow_df_period = streamflow_df.loc[streamflow_start:streamflow_end]

#%% Calculate streamflow data availability per basin (1950–2020)
streamflow_availability = pd.DataFrame({
    'Non-NaN Count': streamflow_df_period.notna().sum(),
    'Availability (%)': 100 * streamflow_df_period.notna().sum() / len(streamflow_df_period)
})

# Sort basins by descending availability
streamflow_availability = streamflow_availability.sort_values(by='Availability (%)', ascending=False)

# Identify basins with 100% data availability
full_streamflow_basins = streamflow_availability[streamflow_availability['Availability (%)'] == 100]
full_streamflow_basin_ids = full_streamflow_basins.index.tolist()

print(f"Number of basins with full streamflow data ({streamflow_start} to {streamflow_end}): {len(full_streamflow_basin_ids)}")

# Subset dataframe to only basins with full streamflow data
streamflow_full_df = streamflow_df_period[full_streamflow_basin_ids]
print("\nSubset of basins with complete streamflow data:")
print(streamflow_full_df.head())

#%% Load meteorology data and check missingness

# Meteorology data folder
meteorology_folder = r"D:\VT_SPR2025\CEE5984_Machine Learning in Water Resources\GroupProject\01_EStreams_Data\EStreams\meteorology"

#%% Function to calculate missing data percentages
def calculate_missingness(basin_list, start_date=None, end_date=None):
    missingness_summary = {}
    
    for basin_id in basin_list:
        file_path = os.path.join(meteorology_folder, f"estreams_meteorology_{basin_id}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: Meteorology file not found for basin {basin_id}")
            continue
        
        # Read meteorology data
        meteo_df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        
        # Subset to specified period if provided
        if start_date and end_date:
            meteo_df = meteo_df.loc[start_date:end_date]
        
        # Calculate missing percentage
        missing_percentage = (meteo_df.isna().sum() / len(meteo_df)) * 100
        missingness_summary[basin_id] = missing_percentage
    
    # Return as DataFrame
    return pd.DataFrame(missingness_summary).T

#%% Calculate missing meteorology data for full streamflow basins (full record)
missing_meteo_full_df = calculate_missingness(full_streamflow_basin_ids)

print("\nMissing meteorology data (% missing) for full streamflow basins (full record):")
print(missing_meteo_full_df)

#%% Calculate missing meteorology data for 1980–2020 period
meteo_start = "1940-01-01"
meteo_end = "2020-12-31"

missing_meteo_period_df = calculate_missingness(full_streamflow_basin_ids, start_date=meteo_start, end_date=meteo_end)

#%%

print(f"\nMissing meteorology data (% missing) for {meteo_start} to {meteo_end}:")
print(missing_meteo_period_df)

#%% Identify basins with full meteorology data (1980–2020)
basins_with_full_meteo = missing_meteo_period_df[(missing_meteo_period_df == 0).all(axis=1)].index.tolist()

print(f"\nNumber of basins with complete meteorology data ({meteo_start}–{meteo_end}): {len(basins_with_full_meteo)}")
print("Basins with full meteorology data:", basins_with_full_meteo)

#%%
#%% Analyze yearly meteorological data availability for full streamflow basins

# Define the analysis period
analysis_start = "1980-01-01"
analysis_end = "2020-12-31"
years_range = pd.date_range(start=analysis_start, end=analysis_end, freq='Y').year

# Meteorological parameters of interest (example: update if needed)
meteo_parameters = ['p_mean', 't_mean', 't_min', 't_max', 'sp_mean', 'rh_mean', 'ws_mean', 'swr_mean', 'pet_mean']

# Initialize dictionaries to store per-parameter missingness data
yearly_missingness_per_parameter = {param: {} for param in meteo_parameters}

# Loop through each basin
for basin_id in full_streamflow_basin_ids:
    file_path = os.path.join(meteorology_folder, f"estreams_meteorology_{basin_id}.csv")
    
    if not os.path.exists(file_path):
        print(f"Warning: Meteorology file not found for basin {basin_id}")
        continue
    
    # Load the meteorology data
    meteo_df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    
    # Subset to analysis period
    meteo_df = meteo_df.loc[analysis_start:analysis_end]
    
    # Check if all desired parameters exist
    available_parameters = [param for param in meteo_parameters if param in meteo_df.columns]
    
    # Calculate % missing per year for each parameter
    for param in available_parameters:
        # Group by year
        yearly_group = meteo_df[param].groupby(meteo_df.index.year)
        
        # Calculate missing % for each year
        missing_percentage_by_year = yearly_group.apply(lambda x: x.isna().sum() / len(x) * 100)
        
        # Store result
        if basin_id not in yearly_missingness_per_parameter[param]:
            yearly_missingness_per_parameter[param][basin_id] = missing_percentage_by_year

# Create a separate DataFrame for each meteorological parameter
parameter_missingness_dfs = {}

for param, basin_data in yearly_missingness_per_parameter.items():
    # Convert dictionary to DataFrame
    df_missingness = pd.DataFrame(basin_data).T  # basins as rows, years as columns
    df_missingness = df_missingness.reindex(columns=years_range)  # Ensure all years included
    parameter_missingness_dfs[param] = df_missingness

#%% Visualization

#%% Visualization: Heatmaps for % missing data per year for each meteorological parameter


import matplotlib.pyplot as plt
import seaborn as sns

# Create a heatmap for each parameter
for param, missing_df in parameter_missingness_dfs.items():
    plt.figure(figsize=( missing_df.shape[1] * 0.7,  missing_df.shape[0] * 0.4))  
    # Width scales with years, height scales with number of basins

    sns.heatmap(
        missing_df, 
        cmap='coolwarm', 
        vmin=0, vmax=100,  # Fix color scale from 0 to 100% missing
        cbar_kws={'label': '% Missing Data'},
        linewidths=0.0,  # Fine gridlines
        linecolor='gray'
    )
    
    plt.title(f"Heatmap of % Missing Data per Year for {param}", fontsize=18)
    plt.xlabel("Year", fontsize=5)
    plt.ylabel("Basin ID", fontsize=5)

    # Optionally: simplify axis ticks if too many
    # if missing_df.shape[0] > 50:
    #     plt.yticks([], [])  # Hide basin IDs
    # if missing_df.shape[1] > 40:
    #     plt.xticks([], [])  # Hide year labels

    plt.tight_layout()
    plt.show()
#%%
import plotly.express as px
import plotly.io as pio

# Optional: open in browser for better experience in Spyder
pio.renderers.default = 'browser'

def plot_interactive_heatmap(df, title="Interactive Heatmap", color_scale="RdBu_r"):
    import plotly.express as px
    import plotly.io as pio

    # Optional: make it open in browser if using Spyder
    pio.renderers.default = 'browser'

    fig = px.imshow(
        df.values,
        labels=dict(x="Year", y="Basin ID", color="% Missing"),
        x=df.columns,
        y=df.index,
        color_continuous_scale=color_scale,  # ✅ use 'RdBu_r' instead of 'coolwarm'
        zmin=0, zmax=100
    )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Basin ID",
        width=max(1000, len(df.columns) * 20),
        height=max(600, len(df) * 12),
        font=dict(size=12)
    )

    fig.show()

#%%
plot_interactive_heatmap(parameter_missingness_dfs['p_mean'], 
                         title="Interactive Heatmap of % Missing PRCP", 
                         color_scale="RdBu_r")


