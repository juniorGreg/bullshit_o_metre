import os

import pandas as pd
from sqlalchemy import create_engine
import psycopg2

from bullshit_o_metre.bullshit_detector import BullshitDetector

from joblib import dump

from sklearn.model_selection import train_test_split


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


    bsd = BullshitDetector()
    bsd.fit(X_train, y_train)

    score = bsd.score(X_test, y_test)

    print(score)

    dump(bsd, "bsd.joblib", compress=True)
