
import spacy

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

import json

import numpy as np


class BullshitDetector(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, max_features=1000, ngram_range=(1,2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vec = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        self.clf = MultinomialNB()
        #self.clf = LogisticRegression(C=200)
        self.nlp = spacy.load('fr_core_news_sm')
        self.stopwords = self.nlp.Defaults.stop_words


    def clean_text(self, text):
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc if token.pos_ != 'PROPN' and token.is_alpha]
        a_lemmas = [lemma for lemma in lemmas if lemma not in self.stopwords]

        return ' '.join(a_lemmas)


    def fit(self, X, y=None):


        X = X.apply(self.clean_text)

        X_data = self.vec.fit_transform(X)

        X = pd.DataFrame(X_data.toarray(), columns=self.vec.get_feature_names())

        self.clf.fit(X, y)

        return self

    def predict_proba(self, X):
        data = X.apply(self.clean_text)
        X_data = self.vec.transform(X)
        X = pd.DataFrame(X_data.toarray(), columns=self.vec.get_feature_names())
        return self.clf.predict_proba(X)


    def predict(self, X):
        data = X.apply(self.clean_text)
        X_data = self.vec.transform(X)
        X = pd.DataFrame(X_data.toarray(), columns=self.vec.get_feature_names())
        return self.clf.predict(X)
