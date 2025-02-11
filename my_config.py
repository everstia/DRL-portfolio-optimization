import pathlib

# import finrl

import pandas as pd
import datetime
import os

DATA_SAVE_DIR = f"datasets"
TRAINED_MODEL_DIR = f"trained_models"
TENSORBOARD_LOG_DIR = f"tensorboard_log"
RESULTS_DIR = f"results"


## Model Parameters
#A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}

TICKERS = [
    "NESTE.HE",
    "UPM.HE",
    "ICP1V.HE",
    "YIT.HE",
    "OLVAS.HE",
    "MEKKO.HE",
    "BOREO.HE",
    "NDA-FI.HE",
    "SAMPO.HE",
    "ELISA.HE",
]
