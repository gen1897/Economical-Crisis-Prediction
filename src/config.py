from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier

DATASET_NAME = "./data/JSTdatasetR4.xlsx"
SHEET_NAME = "Data"
COUNTRIES = ["Norway", "Switzerland", "Finland",
             "Germany", "Denmark", "Netherlands"]
POST_WAR = True

PARAM_GRID = {
    "sgd": {"loss": ["hinge", "log", "squared_loss", "huber", "epsilon_insensitive"],
            "penalty": ["l2", "l1", "elasticnet"], "max_iter": [500, 1000, 1500],
            "learning_rate": ["optimal", "constant", "adaptive", "invscaling"],
            "eta0": [0.00001, 0.0001, 0.005, 0.1]},

    "rf": {"n_estimators": [50, 100, 300, 500, 1000, 1500], "criterion": ["gini", "entropy"],
           "max_features": ["auto", "sqrt", "log2"], "random_state": [42],
           "class_weight": ["balanced", "balanced_subsample", None]},

    "ert": {"n_estimators": [50, 100, 300, 500, 1000, 1500], "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2"], "random_state": [42],
            "class_weight": ["balanced", "balanced_subsample", None]},

    "xgb":  {"random_state": [42], "sampling_method": ["uniform", "gradient_based"],
             "tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
             "grow_policy": ['depthwise', 'lossguide'],
             "predictor": ["auto", "cpu_predictor", "gpu_predictor"]},

    "mlp": {"solver": ["lbfgs", "adam"], "learning_rate": ["constant", "invscaling", "adaptative"],
            "random_state": [42], "activation": ["identity", "logistic", "tanh", "relu"],
            "alpha": [0.00001, 0.0005]}
}

CLASSIFIERS = {
    "sgd": SGDClassifier(),
    "rf": RandomForestClassifier(),
    "ert": ExtraTreesClassifier(),
    "xgb": XGBClassifier(),
    "mlp": MLPClassifier()
}
