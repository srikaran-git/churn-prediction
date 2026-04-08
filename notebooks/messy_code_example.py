# Paste this exactly and run: black notebooks/messy_code_example.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def trainMyModel(X_train, y_train, NumTrees=100, maxD=5, rs=42):
    model = RandomForestClassifier(
        n_estimators=NumTrees, max_depth=maxD, random_state=rs
    )
    model.fit(X_train, y_train)
    return model


def getPredictions(model, X):
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    return preds, probas


class myMLmodel:
    def __init__(self, modelType, params):
        self.modelType = modelType
        self.params = params
        self.model = None

    def train(self, X, y):
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
