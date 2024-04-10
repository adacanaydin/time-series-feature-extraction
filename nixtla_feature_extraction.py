import os
from tsfeatures import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)

def calculate_data_quality_metrics(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate missing, null percentages, and duration."""
    missing_perc = df.groupby(['unique_id'])['y'].apply(lambda x: float(x.isnull().sum()/ len(x)*100))
    null_perc = df.groupby(['unique_id'])['y'].apply(lambda x: (x == 0.00).astype(int).sum(axis=0)/len(x)*100)
    
    if not isinstance(df['timestamp'], pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    duration = (df.groupby(['unique_id'])['timestamp'].apply(lambda x: (x.max() - x.min()).days)
                .astype(int))

    return missing_perc, null_perc, duration

def visualize_data_quality(missing_perc: pd.Series, null_perc: pd.Series, duration: pd.Series, dataset_name: str):
    """Plot data quality metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot missing percentage
    missing_perc.sort_values(ascending=False).head(25).plot(kind="barh", ax=axes[0])
    axes[0].invert_yaxis()
    axes[0].set_title(f'Top 25 buildings with highest missing percentage ({dataset_name})')
    axes[0].set_xlabel('Missing Percentage (%)')

    # Plot null percentage
    null_perc.sort_values(ascending=False).head(25).plot(kind="barh", ax=axes[1])
    axes[1].invert_yaxis()
    axes[1].set_title(f'Top 25 buildings with highest null percentage ({dataset_name})')
    axes[1].set_xlabel('Null Percentage (%)')

    # Plot duration
    duration.sort_values(ascending=True).head(25).plot(kind="barh", ax=axes[2])
    axes[2].invert_yaxis()
    axes[2].set_title(f'Top 25 buildings with lowest observed timespan ({dataset_name})')
    axes[2].set_xlabel('Observed timespan (days)')

    plt.tight_layout()
    plt.show()

def drop_buildings(df: pd.DataFrame, missing_perc: pd.Series, null_perc: pd.Series, duration: pd.Series) -> pd.DataFrame:
    """Drop buildings with more than 50% missing values, more than 99% null values, or less than 30 days of readings."""
    index_missing = missing_perc[missing_perc >= 50].index 
    index_null = null_perc[null_perc >= 99].index  
    index_duration = duration[duration < 30].index
    indices = index_duration.append([index_null, index_missing]).drop_duplicates().tolist()
    logging.info("Buildings are dropped based on data quality metrics.")
    return df[~df['unique_id'].isin(indices)]

def impute_missing_with_weekly_median(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with the median values of the daily usage profile of each day of the week for each household."""
    if df['y'].isnull().any():
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['day_of_week'] = df['timestamp'].dt.day_name()
        median_profiles = df.groupby(['unique_id', 'day_of_week'])['y'].transform('median')
        df['y'] = df['y'].fillna(median_profiles)
        df.drop(columns=['day_of_week'], inplace=True)
        logging.info('Missing values are imputed.')
    else:  
        logging.info('There is no need for imputation.')
    
    return df

def detect_consecutive_days_with_same_value(df: pd.DataFrame) -> pd.Index:
    """Detect consecutive days with the same value."""
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby(['unique_id', 'date'])['y'].nunique()
    consecutive_days = daily_counts[daily_counts == 1]
    indices_to_remove = df[df.set_index(['unique_id', 'date']).index.isin(consecutive_days.index)].index
    return indices_to_remove

def process_data(df: pd.DataFrame, dataset_name: str) -> None:
    """Process data for a given dataset."""
    # Calculate data quality metrics
    missing_perc, null_perc, duration = calculate_data_quality_metrics(df)
    # Visualize data quality
    visualize_data_quality(missing_perc, null_perc, duration, dataset_name)
    # Drop buildings based on data quality metrics
    df_processed = drop_buildings(df, missing_perc, null_perc, duration)
    # Impute missing values
    df_processed = impute_missing_with_weekly_median(df_processed)
    # Remove the rows corresponding to the consecutive days with the same value for each unique ID 
    indices_to_remove = detect_consecutive_days_with_same_value(df_processed)
    df_processed = df_processed.drop(indices_to_remove)
    logging.info("Rows corresponding to days with the consecutive same value are removed.")

def extract_features(path: str, dataset_name: str, freq: int) -> pd.DataFrame:
    """Extract features for a given dataset."""
    logging.info(f"Reading data from file: {path}")
    if dataset_name == 'fluvius':
        df = pd.read_csv(path, delimiter=';', encoding='utf-8')
        df = df.reset_index(drop=True)
        logging.info("Performing additional transformations for 'fluvius' dataset...")
        df.rename(columns={'EAN_ID': 'unique_id', 'Datum_Startuur': 'timestamp', 'Volume_Afname_kWh': 'y'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'].str[:-1], format='%Y-%m-%dT%H:%M:%S.%f')
        df.drop(['Volume_Injectie_kWh', 'Warmtepomp_Indicator', 'Elektrisch_Voertuig_Indicator', 'PV-Installatie_Indicator', 'Datum', 'Contract_Categorie', 'Class_Name'], axis=1, inplace=True)
        df
    else:
        df = pd.read_parquet(path)
        df = df.reset_index(drop=True)

    if 'stdorToU' in df.columns:
        df = df.drop('stdorToU', axis=1)
    if 'ds' in df.columns:
        df = df.rename(columns={"ds": "timestamp"})
    if 'LCLid' in df.columns:
        df = df.rename(columns={"LCLid": "unique_id"})
    if 'DateTime' in df.columns:
        df = df.rename(columns={"DateTime": "timestamp"})
    if 'KWH/hh (per half hour) ' in df.columns:
        df = df.rename(columns={"KWH/hh (per half hour) ": "y"})
    #df['unique_id'] = pd.to_numeric(df['unique_id'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit="s")
    df['y'] = pd.to_numeric(df['y'])
    logging.info(df.info())
    logging.info("Pre-processing the dataset...")
    process_data(df, dataset_name)
    logging.info("Performing feature extraction...")
    features = tsfeatures(df, freq=freq)
    logging.info("Feature extraction completed.")
    return features

def process_features(dataset_name: str, path: str, freq: int, output_path: str) -> None:
    """Process features for a given dataset."""
    logging.info(f"Processing {dataset_name} features...")
    features = extract_features(path, dataset_name, freq)
    features.to_parquet(output_path)
    logging.info(f"{dataset_name} features processed successfully.")

if __name__ == "__main__":
    process_features("London", "./data/raw/london/CC_LCL-FullData.parquet", 48, "./data/features/nixtla_features_london.parquet")
    process_features("Australia", "./data/raw/australia/Australia.parquet", 48, "./data/features/nixtla_features_australia.parquet")
    process_features("BDG2", "./data/raw/bdg2/BDG2_clean.parquet", 24, "./data/features/nixtla_features_bdg2.parquet")
    process_features("CHP", "./data/raw/chp/clean_production.parquet", 24, "./data/features/nixtla_features_chp.parquet")
    ###process_features("Ireland", "./data/raw/", ??, "./data/features/nixtla_features_irish.parquet")
    process_features("Fluvius", "./data/raw/fluvius/P6269_1_50_DMK_Sample_Elek.csv", 96, "./data/features/nixtla_features_fluvius.parquet")


