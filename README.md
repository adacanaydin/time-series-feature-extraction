# Time Series Feature Extraction

This repository contains scripts for extracting time series features using two methods: Nixtla features (domain-agnostic) and IFEEL features (domain-informed).

## Overview

**Nixtla Features (Domain-Agnostic):** Extracts a set of features from time series data that are domain-agnostic, suitable for a wide range of applications.

**IFEEL Features (Domain-Informed):** Extracts features from time series data that are informed by the domain, providing insights tailored to daily load profiles.

## Usage

### 1. Nixtla Feature Extraction

This script contains functions to process time series data and extract features using the Nixtla library. It performs the following tasks:

#### 1.1 Data Quality Metrics Calculation

Calculate missing percentages, null percentages, and duration for each unique ID in the dataset.

#### 1.2 Data Quality Visualization

Visualize missing percentages, null percentages, and duration using bar plots.

#### 1.3 Data Preprocessing

- Drop buildings with more than 50% missing values, more than 99% null values, or less than 30 days of readings.
- Impute missing values with weekly median values.
- Remove rows corresponding to consecutive days with the same value for each unique ID.

#### 1.4 Feature Extraction

Extract features using the Nixtla library, such as statistical, frequency, and temporal features. Save the extracted features as a Parquet file.

To use the script, execute it with Python:
python nixtla_feature_extraction.py


### 2. IFEEL Feature Extraction

This script contains functions to process time series data and extract features using the IFEEL library. It performs the following tasks:

#### 2.1 Data Preprocessing

- Resample the data to a specified frequency.
- Interpolate missing values.

#### 2.2 Feature Extraction

Extract global features for each daily profile, including statistical and temporal features. Extract peak period features for each daily profile. Save the extracted features as CSV files.

To use the script, execute it with Python:
python ifeel_feature_extraction.py


## Dependencies

- Python 3
- pandas
- seaborn
- matplotlib
- Nixtla
- IFEEL

Install the required Python packages using pip:
pip install -r requirements.txt


