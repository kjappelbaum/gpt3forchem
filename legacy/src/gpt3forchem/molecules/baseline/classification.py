from sklearn.linear_model import LogisticRegressionCV

import numpy as np
from gpt3forchem.baseline import BaseLineModel


class LogisticClassifierBaseline(BaseLineModel):
    def __init__(self, seed) -> None:
        self.model = LogisticRegressionCV(cv=5, random_state=seed)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def tune(self, X_train, y_train):
        pass
