from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# ================================!!!!!!!!!================================
# Create a new binary feature that is positive two years after a crisis event for a given country.


def precrisis(df: pd.DataFrame):
    # Create feature.
    df["precrisis"] = 0

    # Set value to one on both previous years when a crisis is given.
    for i in range(1, len(df)):
        if (df.loc[i, "crisisJST"] == 1):
            df.loc[(i-1), "precrisis"] = 1
            df.loc[(i-2), "precrisis"] = 1

    return df
# ================================!!!!!!!!!================================
# Drop the crisis and the next 5 samples to avoid a possible bias given in the recovery stage.


def drop_crisis_bias(df: pd.DataFrame):
    # Feature positive in the years that have to be dropped.
    df["drops"] = 0

    # Set values to positive.
    for i in range(len(df)):
        if (df.loc[i, "crisisJST"] == 1):
            df.loc[i, "drops"] = 1
            df.loc[i + 1, "drops"] = 1
            df.loc[i + 2, "drops"] = 1
            df.loc[i + 3, "drops"] = 1
            df.loc[i + 4, "drops"] = 1
            df.loc[i + 5, "drops"] = 1

    # Drop biased samples.
    df = df[df["drops"] == 0]
    # Drop the new feature.
    df = df.drop(["drops"], axis=1)

    return df
# ================================!!!!!!!!!================================
# Drop biases years such as wars or Great depression


def drop_years(df: pd.DataFrame):
    excluded_years = [1914, 1915, 1916, 1917, 1918, 1933, 1934, 1935,
                      1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945]
    df = df[~df["year"].isin(excluded_years)]

    return df
# ================================!!!!!!!!!================================
# Select desired countries


def select_countries(df: pd.DataFrame, countries: list):
    df = df[df["country"].isin(countries)]

    return df
# ================================!!!!!!!!!================================
# Drop missing values


def drop_null(df: pd.DataFrame):
    # Convert infinite with NaN
    df = df.replace(np.inf, np.NaN)
    # Drop values
    df = df.dropna()

    return df
# ================================!!!!!!!!!================================
# Create new features


def create_features(df: pd.DataFrame, periods: int):
    # Slope of the Yield Curve
    df["slope"] = df["ltrate"] - df["stir"]

    # Global Credit
    df["global_credit"] = 0
    for i in range(len(df)):
        # Get the year
        year_it = df.loc[i, "year"]
        # Mean credit of all countries excepts the one
        df["global_credit"][i] = (df[df["year"] == year_it].tloans.sum(
        ) - df.loc[i, "tloans"]) / (len(df[df["year"] == year_it].tloans) - 1)
    # Get growth of global credit
    df["gr_global_credit"] = df["global_credit"].pct_change(periods=periods)

    # Logarithm of Consumer Price Index and its growth
    df["log_cpi"] = np.log(df["cpi"])
    df["gr_log_cpi"] = df["log_cpi"].pct_change(periods=periods)

    # Current Account scaled by GDP
    df["ca_gdp"] = df["ca"]/df["gdp"]
    df["gr_ca_gdp"] = df["ca_gdp"].pct_change(periods=periods)

    # Log RGDP
    df["log_real_gdp"] = np.log(df["rgdpmad"])
    df["gr_log_real_gdp"] = df["log_real_gdp"].pct_change(periods=periods)

    # Log broad money
    df["log_money"] = np.log(df["money"])
    df["gr_log_money"] = df["log_money"].pct_change(periods=periods)

    # Log domestic credit
    df["log_credit"] = np.log(df["tloans"])
    df["gr_log_credit"] = df["log_credit"].pct_change(periods=periods)

    # Inversion
    df["gr_inv"] = df["iy"].pct_change(periods=periods)

    # GDP growth
    df["gr_gdp"] = df["gdp"].pct_change(periods=periods)

    return df
# ================================!!!!!!!!!================================
# Find better hyperparameters and return score


def find_parameters(clf, param_grid: dict, X, y, scoring: str = "roc_auc", cv: int = 5, return_train: bool = True):
    grid_search = GridSearchCV(
        clf, param_grid, scoring=scoring, cv=cv, return_train_score=return_train)

    grid_search.fit(X, y)

    return {"best_params": grid_search.best_params_, "score": grid_search.best_score_}
