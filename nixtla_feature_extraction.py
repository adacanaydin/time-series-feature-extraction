import os
import warnings
from tsfeatures import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from pathlib import Path

def import_data_chp(csv_file_path: Path):
    """Import the CHP data from a csv file

    Args:
        csv_file_path (the path of the csv file): this is already provided in the __init__() as ../Data/

    Returns:
        chp (DataFrame): A Dataframe with, for each CHP, the output, forecasted output and capacity
        production (DataFrame): A Dataframe containing only the output for each CHP, to be used for our models
    """
    # Read the CSV file
    chp = pd.read_csv(csv_file_path, index_col='Date', low_memory=False)
    # Create a multi-index column
    header_values = chp.columns
    locations = header_values[0::3]
    metrics = ['Installed Capacity NCNR MW', 'NCNR production Forecast IntraDay MW', 'NCNR production MW']
    header = pd.MultiIndex.from_product([locations, metrics])
    chp.columns = header
    chp = chp[1:]
    chp = chp.astype(float)  # The values are all read as strings, so we need to adjust all of them to floats
    chp.drop(columns=['1680'], inplace=True)  # This CHP has no output for the entire period
    # Create a new datetime index
    new_datetime_index = pd.date_range(start='01-01-2021 00:00:00', end='13-05-2023 23:00:00', freq='H')
    # Reindex the DataFrame with the new datetime index
    chp.index = new_datetime_index
    # Get only the production values
    production = chp.xs('NCNR production MW', level=1, axis=1)

    return chp, production

def preprocess_and_plot(production):
    """Preprocess the data and generate missing value plots."""
    production.index.name = 'timestamp'
    production.columns.names = ['unique_id']

    df_plot = production.resample('D').mean()
    df_plot.index = df_plot.index.astype('str')
    df_plot = ~df_plot.isna().T
    df_plot = df_plot.sort_index()

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(df_plot, vmin=0, vmax=1, cbar=False)
    ax.set_title('Missing data in CHP dataset.')

    image_path = os.path.join(images_dir, 'missing_data_heatmap_chp.png')
    plt.savefig(image_path)

    # Format the data into the Nixtla format and drop buildings 
    production_reset = production.reset_index()
    melted_production = pd.melt(production_reset, id_vars=['timestamp'], value_vars=production_reset.columns[1:], var_name='unique_id', value_name='y')
    melted_production['timestamp'] = pd.to_datetime(melted_production['timestamp'])
    melted_production = melted_production[['unique_id', 'timestamp', 'y']]

    # Filter out rows with specified unique_id values
    exclude_ids = [1680, 31574, 3859, 34]
    melted_production = melted_production[~melted_production['unique_id'].isin(exclude_ids)]

    # Remove unique_id with more than 40% missing values
    missing_threshold = 40
    missing_perc = pd.DataFrame(melted_production.groupby(['unique_id'])['y'].apply(lambda x: float(x.isnull().sum() / len(x) * 100))).rename({'y': 'missing_percentage'}, axis=1)
    index_missing = missing_perc[missing_perc.missing_percentage >= missing_threshold].index.tolist()
    melted_production = melted_production.loc[~melted_production['unique_id'].isin(index_missing)]

    # Remove rows corresponding to days with all hours having 'y' value of 0, and/or with all hours having same 'y' value
    melted_production['timestamp'] = pd.to_datetime(melted_production['timestamp'])
    melted_production['date'] = melted_production['timestamp'].dt.date

    days_with_all_same = melted_production.groupby(['unique_id', 'date'])['y'].nunique().eq(1).reset_index()
    days_all_same = days_with_all_same[days_with_all_same['y']]

    filtered_production = melted_production[~melted_production.set_index(['unique_id', 'date']).index.isin(days_all_same.set_index(['unique_id', 'date']).index)]
    filtered_production = filtered_production.drop(columns=['date'])

    # Impute missing values
    columns_to_impute = ['y']
    imputer = KNNImputer()

    def impute_group(group):
        group[columns_to_impute] = imputer.fit_transform(group[columns_to_impute])
        return group

    imputed_production = filtered_production.groupby('unique_id').apply(impute_group)
    imputed_production.reset_index(drop=True, inplace=True)

    return imputed_production

def save_clean_production(imputed_production, output_dir:str):
    """Save the clean version of the dataset."""
    output_path = os.path.join(output_dir, 'clean_production.parquet')
    imputed_production.to_parquet(output_path)

def extract_chp_features(dataset: pd.DataFrame):
    """Extract Nixtla features from time series data."""

    warnings.filterwarnings("ignore")

    file_path = Path(f"./data/features/nixtla_features_chp.parquet")

    if file_path.exists():
        print(f"Dataset {dataset} has been processed already.")
        return

    print(f"Processing dataset: {dataset}")
    try:
        features = tsfeatures(dataset, freq=24)

        # Save features
        print(f"Extracted {len(features)} from site {dataset}.")
        features.to_parquet(file_path, index=False)
      
    except Exception as e:
        print(f"Error happened while feature extraction for dataset {dataset}, {e}")
        raise e
    
def main():
    # Import data
    csv_file_path = Path("./data/raw/chp/CHPsData2021_2023.csv")
    chp, production = import_data_chp(csv_file_path)

    # Preprocess data and generate plots
    imputed_production = preprocess_and_plot(production)

    # Save clean production data
    output_dir = Path("./data/raw/chp/")
    save_clean_production(imputed_production, output_dir)

    print("Preprocessing and saving completed.")

    # Extract Nixtla features
    extract_chp_features(imputed_production)

    print("Feature extraction and saving completed.")


if __name__ == "__main__":
    # Create images directory if it doesn't exist
    images_dir = Path("./images/")
    images_dir.mkdir(parents=True, exist_ok=True)

    main()
