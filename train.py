import os

import pandas as pd
from sqlalchemy import create_engine
import psycopg2

from bullshit_o_metre.bullshit_detector import BullshitDetector

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import numpy as np

from hyperopt import hp, fmin, tpe


if __name__ == "__main__":

    database_url = os.environ["DATABASE_URL"]
    sql_query = os.environ["SQL_QUERY"]

    alchemyEngine = create_engine(database_url, pool_recycle=3600)
    #mlflow.log_param("database_url", database_url)

    dbConnection =  alchemyEngine.connect()

    dataframe = pd.read_sql(sql_query, dbConnection)

    print(len(dataframe))

    X = dataframe['texts']

    y = dataframe['is_bullshit'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    def combinations_ngram_range(max_range):
        ngram_ranges = []
        numbers = list(range(1, max_range+1))
        for number_a in numbers:
            ngram_ranges.append((1, number))

        print(ngram_ranges)
        return ngram_ranges


    space = {
        'max_features': hp.quniform("max_features", 450, 3000, 25),
        'ngram_range': hp.quniform("ngram_range", 1, 10, 1)
    }

    def objective(params):
         bsd = BullshitDetector(max_features=int(params["max_features"]), ngram_range=(1,int(params['ngram_range'])))
         bsd.fit(X_train, y_train)

         loss = 1 - bsd.score(X_test, y_test)
         return loss

    best = fmin(fn=objective, space=space, max_evals=5, rstate=np.random.RandomState(42), algo=tpe.suggest)
    print(best)

    bsd = BullshitDetector(max_features=int(best["max_features"]), ngram_range=(1,int(best["ngram_range"])))
    bsd.fit(X_train, y_train)

    print(bsd.score(X_test, y_test))

    dump(bsd, "bsd.joblib", compress=True)
