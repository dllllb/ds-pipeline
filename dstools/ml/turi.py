from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

import turicreate as tc
import numpy as np


class SklearnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, constructor=tc.classifier.create, output_type='probability', **params):
        self.constructor = constructor
        self.params = params
        self.target = 'target'
        self.output_type = output_type
        self.model = None
        pass

    def fit(self, X, y):
        self.sf = tc.SFrame(X)
        self.sf['target'] = y
        self.n_classes = len(self.sf['target'].unique())
        self.model = self.constructor(dataset=self.sf, target=self.target, **self.params)
        return self

    def predict(self, X):
        return np.array(self.model.predict(tc.SFrame(X)))

    def predict_proba(self, X):
        scores = self.model.predict_topk(tc.SFrame(X),
                                         output_type=self.output_type,
                                         k=self.n_classes).to_dataframe()
        id = 'id' if 'id' in scores.columns else 'row_id'
        scores[id] = scores[id].astype(int)
        y_score = scores.pivot(index=id, columns='class', values=self.output_type)
        return y_score.as_matrix()

    def get_params(self, deep=True):
        if self.model is None:
            return self.params
        return self.model.get_current_options()
