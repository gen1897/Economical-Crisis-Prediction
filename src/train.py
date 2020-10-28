import pandas as pd
import time

import utils
from config import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def preprocess(df: pd.DataFrame, countries: list, periods: int = 2):
    # Add precrisis feature
    df = utils.precrisis(df)
    # Add new features
    df = utils.create_features(df, periods)
    # Remove post-crisis bias
    df = utils.drop_crisis_bias(df)
    # Remove biases years
    df = utils.drop_years(df)
    # Reduce the number of columns to minimize the number of samples to be dropped
    df = df[["precrisis", "slope", "gr_global_credit", "gr_log_cpi", "ca_gdp", "gr_log_real_gdp", "gr_log_money",
             "gr_log_credit", "gr_inv", "year", "country", "stir", "gr_ca_gdp", "gr_gdp"]]
    # Drop missing values
    df = utils.drop_null(df)
    # Select desired countries
    df = utils.select_countries(df, countries)

    return df


# Read dataset
JST_df = pd.read_excel(DATASET_NAME, sheet_name=SHEET_NAME)

# Prepare dataset
df = preprocess(JST_df, COUNTRIES)

# Select only years after WWII
if POST_WAR:
    df = df[df["year"] > 1949]

# Features and target
X = df[["slope", "gr_global_credit", "gr_log_cpi", "ca_gdp", "gr_log_real_gdp", "gr_log_money",
        "gr_log_credit", "gr_inv"]]
y = df["precrisis"]

# Initialize scores and best parameters dicts
scores = {}
best_params = {}
times = {}

# Fill dicts with the best parameters and its scores
for key in CLASSIFIERS:
    start_time = time.time()

    result = utils.find_parameters(
        CLASSIFIERS[key], PARAM_GRID[key], X, y)
    scores[key] = result["score"]
    best_params[key] = result["best_params"]

    total_time = time.time() - start_time
    times[key] = total_time

    print("Scores:", scores)
    print("Best parameters:", best_params)
    print("Time:", total_time)

# Report results into a .txt
with open("report.txt", "a") as f:
    f.write("Scores: \n \t {}\nResults: \n \t {}\nTimes (sec): \n \t {} \n".format(
        scores, best_params, times))
