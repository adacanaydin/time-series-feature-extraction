# Time Series Feature Extraction
This repository contains scripts for extracting time series features using two methods: Nixtla features (domain-agnostic) and IFEEL features (domain-informed).

# Overview
Nixtla Features (Domain-Agnostic): Extracts a set of features from time series data that are domain-agnostic, suitable for a wide range of applications.

IFEEL Features (Domain-Informed): Extracts features from time series data that are informed by the domain, providing insights tailored to daily load profiles.

# Usage
1. Nixtla Feature Extraction:

Run nixtla_feature_extraction.py to preprocess and clean the data for each dataset. At the moment, preprocessing steps are provided for the CHP dataset. Soon, common preprocessing steps for different datasets will be added.
The script also extracts Nixtla features from the preprocessed data.

2. IFEEL Feature Extraction:

Run ifeel_feature_extraction.py to obtain domain-informed features from the preprocessed data.