import pymongo
import pandas as pd
import json
from dataclasses import dataclass
import os
# Provide the mongodb localhost url to connect python to mongodb.

@dataclass()
class EnvironmentVariable:
    mongo_db_url=os.getenv("Mongo_DB_URL")

env_var=EnvironmentVariable()


TARGET_COLUMN = "target"
feature_cols = ["age",
                "sex",
                "on_thyroxine",
                "query_on_thyroxine",
                "on_antithyroid_medication",
                "sick",
                "pregnant",
                "thyroid_surgery",
                "I131_treatment",
                "query_hypothyroid",
                "query_hyperthyroid",
                "lithium",
                "goitre",
                "tumor",
                "hypopituitary",
                "psych",
                "TSH measured",
                "TSH",
                "T3_measured",
                "T3",
                "TT4_measured",
                "TT4",
                "T4U_measured",
                "T4U",
                "FTI_measured",
                "FTI",
                "TBG_measured",
                "TBG",
               "target"]
feature_index = ["on_thyroxine",
                "query_on_thyroxine",
                "on_antithyroid_medication",
                "sick",
                "pregnant",
                "thyroid_surgery",
                "I131_treatment",
                "query_hypothyroid",
                "query_hyperthyroid",
                "lithium",
                "goitre",
                "tumor",
                "hypopituitary",
                "psych",
                "sex",
                "age",
                "TSH",
                "T3",
                "TT4",
                "T4U",
                "FTI",
               "target"]
numerical_columns=['TSH','T3','TT4','T4U','FTI']
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)