import warnings
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from src_ifeel import  ifeel_extraction
from src_ifeel import ifeel_transformation
import sys
import numpy as np

def process_site(
    dataset: str,
    unique_id: str,
    site_df: pd.DataFrame,
    freq: str,
    time_business_start=9,
    time_business_end=17,
    alphabet_size=7,
):
    warnings.filterwarnings("ignore")

    file_path = Path(f"./data/features/{dataset}/ifeel_features_{unique_id}.csv")  
    
    
    if file_path.exists():
        print(f"Site {unique_id} has been processed already.")
        return

    print(f"Processing site: {unique_id}")
    try:
        # Drop the duplicated values for the same time per individual buildings
        site_df = site_df.drop_duplicates(subset=['unique_id', 'timestamp'], keep='first')
        site_df = site_df.set_index("timestamp").resample(freq).asfreq()
        # Get the date and time info into separate columns for future processing
        site_df['time'] = site_df.index.time
        site_df['date'] = site_df.index.date
        site_df = site_df.interpolate(limit=1)
        # Format the data into the IFEEL format and drop non-full days
        _ifeel_format = site_df.pivot(index='date', columns='time', values='y').dropna(axis=0, how="any")
        _ifeel_format.columns = [c.strftime('%H:%M:%S') for c in
                                 _ifeel_format.columns]  # I feel that IFEEL cannot handle time

        sample_interval_in_hour = 24 / _ifeel_format.shape[1]
        # note: the value of sample interval is in the unit of hour, e.g., if the interval is 30 minutes, then sample_interval = 0.5.

        # Data transformation
        [df_raw, df_raw_diff, df_SAX_number, _, df_SAX_number_diff] = ifeel_transformation.feature_transformation(
            _ifeel_format, alphabet_size, time_business_start, time_business_end
        )

        # Global feature extraction for each daily profile
        feature_global_all_days = pd.DataFrame()
        for i in np.arange(0, df_raw.shape[0]):
            ts = df_raw.iloc[i]
            ts_diff = df_raw_diff.iloc[i]
            feature_global_all_each = ifeel_extraction.feature_global(
                ts, ts_diff, sample_interval_in_hour).global_all().T
            feature_global_all_days = pd.concat([feature_global_all_days,feature_global_all_each], axis=0, ignore_index=True)
        feature_global_all_days.columns = ifeel_extraction.feature_name_global

        # Peak feature extraction for each daily profile
        feature_peak_period_all_days = pd.DataFrame()
        for i in np.arange(0, df_raw.shape[0]):
            ts_sax = df_SAX_number.iloc[i]
            ts_sax_diff = df_SAX_number_diff.iloc[i]
            feature_peak_all_each = ifeel_extraction.feature_peak_period(
                ts_sax, ts_sax_diff, alphabet_size, sample_interval_in_hour).T
            feature_peak_period_all_days = pd.concat([feature_peak_period_all_days,feature_peak_all_each], axis=0, ignore_index=True)
        feature_peak_period_all_days.columns = ifeel_extraction.feature_name_peak

        site_features_all_days = pd.concat([feature_global_all_days, feature_peak_period_all_days], axis=1)
        site_features_all_days['date'] = _ifeel_format.index
        site_features_all_days['unique_id'] = unique_id
        # site_features_all_days['Peak_all: time'] = site_features_all_days['Peak_all: time'].apply(
        #     lambda x: ';'.join(map(str, x)))
        # site_features_all_days['Peak_all: duration'] = site_features_all_days['Peak_all: duration'].apply(
        #     lambda x: ';'.join(map(str, x)))

        # Save features
        print(f"Extracted {len(site_features_all_days)} from site {unique_id}.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        site_features_all_days.to_csv(file_path, index=False)
      
    except Exception as e:
        print(f"Error happened while processing site {unique_id}, {e}")
        raise e


def process_one_by_one(df: pd.DataFrame, dataset: str, freq: str):
    # Run them one by one takes forever
    for unique_id, group in df.groupby("unique_id"):
        process_site(dataset, str(unique_id), group, freq)


def process_parallel(df: pd.DataFrame, dataset: str, freq: str):
    # Run them in parallel, still takes a long time
    with Pool() as pool:
        args = [(dataset, unique_id, group, freq,) for unique_id, group in df.groupby("unique_id")]
        pool.starmap_async(process_site, args).wait()


def extract_irish_features(path: str, parallel: bool):
    df = pd.read_parquet(path)
    # Convert them to numeric
    df['unique_id'] = pd.to_numeric(df['unique_id'])
    df['time_code'] = pd.to_numeric(df['time_code'])
    df['y'] = pd.to_numeric(df['y'])
    # Take the last 2 values and subtract 1 (goes from 1 to 48 and not from 0 to 47)
    df['_time_code'] = (df['time_code'] % 100).astype(int) - 1
    # Drop the days with more than 48 values, some of them are due to daylight saving time, but makes it simpler
    df = df.loc[df['_time_code'] < 48]
    # Take the first values without the last 2
    df['_day_code'] = (df['time_code'] / 100).apply(np.floor).astype(int)
    # Convert the day and time code to timestamps
    start_unix = int(datetime.fromisoformat("2007-01-01T00:00:00+00:00").timestamp())
    df['unix'] = start_unix + df['_day_code'] * 24 * 60 * 60 + df['_time_code'] * 30 * 60
    df['timestamp'] = pd.to_datetime(df['unix'], unit="s")
    print(df.info())

    if parallel:
        process_parallel(df, "irish", "30T")
    else:
        process_one_by_one(df, "irish", "30T")


def extract_london_features(path: str, parallel: bool):
    df = pd.read_parquet(path)
    df = df.rename(
        columns={
            'KWH/hh (per half hour) ': "y",
            "DateTime": "timestamp",
            "LCLid": "unique_id",
        }
    )
    print(df.info())

    if parallel:
        process_parallel(df, "london", "30T")
    else:
        process_one_by_one(df, "london", "30T")


def extract_australia_features(path: str, parallel: bool):
    df = pd.read_parquet(path)

    if parallel:
        process_parallel(df, "australia", "30T")
    else:
        process_one_by_one(df, "australia", "30T")


def extract_bdg2_features(path: str, parallel: bool):
    df = pd.read_parquet(path)
    df = df.rename(
        columns={
            "ds": "timestamp",
        }
    )
    print(df.info())

    if parallel:
        process_parallel(df, "bdg2", freq="1H")
    else:
        process_one_by_one(df, "bdg2", freq="1H")
        
def extract_chp_features(path: str, parallel: bool):
    df = pd.read_parquet(path)
    df['unique_id'] = pd.to_numeric(df['unique_id'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit="s")
    df['y'] = pd.to_numeric(df['y'])

    print(df.info())

    if parallel:
        process_parallel(df, "chp", freq="1H")
    else:
        process_one_by_one(df, "chp", freq="1H")

def extract_fluvius_features(path: str, parallel: bool):
    df = pd.read_parquet(path)

    if parallel:
        process_parallel(df, "fluvius",freq="15T")
    else:
        process_one_by_one(df, "fluvius", freq="15T")


def merge_features(dataset: str) -> pd.DataFrame:
    files = glob(f"./data/features/{dataset}/ifeel_features_*.csv")
    return pd.concat([pd.read_csv(file) for file in files]).reset_index()


def process_irish_features(parallel: bool):
    extract_irish_features("./data/raw/irish/File1.parquet", parallel)
    extract_irish_features("./data/raw/irish/File2.parquet", parallel)
    extract_irish_features("./data/raw/irish/File3.parquet", parallel)
    extract_irish_features("./data/raw/irish/File4.parquet", parallel)
    extract_irish_features("./data/raw/irish/File5.parquet", parallel)
    extract_irish_features("./data/raw/irish/File6.parquet", parallel)
    merge_features("irish").to_parquet("./data/features/ifeel_features_irish.parquet")


def process_london_features(parallel: bool):
    extract_london_features("./data/raw/london/CC_LCL-FullData.parquet", parallel)
    merge_features("london").to_parquet("./data/features/ifeel_features_london.parquet")


def process_australian_features(parallel: bool):
    extract_australia_features("./data/raw/australia/Australia.parquet", parallel)
    merge_features("australia").to_parquet("./data/features/ifeel_features_australia.parquet")


def process_bdg2_features(parallel: bool):
    extract_bdg2_features("./data/raw/bdg2/BDG2_clean.parquet", parallel)
    merge_features("bdg2").to_parquet("./data/features/ifeel_features_bdg2.parquet")
  
def process_chp_features(parallel: bool):
    extract_chp_features("./data/raw/chp/clean_production.parquet", parallel)  
    merge_features("chp").to_parquet("./data/features/ifeel_features_chp.parquet")    

def process_fluvius_features(parallel: bool):
    extract_fluvius_features("./data/raw/fluvius/fluvius.parquet", parallel)  
    merge_features("fluvius").to_parquet("./data/features/ifeel_features_fluvius2.parquet")   

if __name__ == '__main__':
    # Process features, parallel=False/True
    process_irish_features(True)
    process_london_features(True)
    process_australian_features(True)
    process_bdg2_features(True)
    process_chp_features(True) 
    process_fluvius_features(True)
