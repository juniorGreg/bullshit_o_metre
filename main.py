from bullshit_o_metre import bullshit_detector
from flask import Flask, request

from joblib import load
import pandas as pd

application = Flask(__name__)
bsd = load("bsd.joblib")

@application.route("/")
def main():
    return "HELLLLLO!, Bullshit-o-metre api!!!!"

@application.route("/predict", methods=["POST"])
def predict():
    data = request.data.decode("utf8")
    query_df = pd.Series([data])

    preds = bsd.predict_proba(query_df)

    print(preds)

    return "{:.2f}".format(preds[0][1] * 100)



if __name__ == '__main__':
    app.run()
