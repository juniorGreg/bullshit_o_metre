import os

import pandas as pd
from sqlalchemy import create_engine
import psycopg2

from bullshit_o_metre.bullshit_detector import BullshitDetector

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import numpy as np


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
            for number_b in numbers[(number_a-1):]:
                ngram_ranges.append((number_a, number_b))

        print(ngram_ranges)
        return ngram_ranges


    param_grid = {'max_features': list(range(450, 3001)), 'ngram_range': combinations_ngram_range(6)}

    random_bsd =RandomizedSearchCV(
        estimator = BullshitDetector(),
        param_distributions = param_grid,
        n_iter = 10,
        scoring='accuracy',
        n_jobs=2,
        cv=5,
        refit=True,
        return_train_score=True
    )


    random_bsd.fit(X_train, y_train)

    print(random_bsd.cv_results_['param_max_features'])
    print(random_bsd.cv_results_['param_ngram_range'])

    print(random_bsd.best_estimator_.score(X_test, y_test))

    dump(random_bsd.best_estimator_, "bsd.joblib", compress=True)
