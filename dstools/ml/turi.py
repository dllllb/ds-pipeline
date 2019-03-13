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


def roc_auc_avg_score(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing.label import label_binarize

    y_bin = label_binarize(y_true, classes=sorted(set(y_true)))
    return roc_auc_score(y_bin, y_score)


def test_turi_skearn():
    from sklearn.datasets import load_iris
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score

    iris = load_iris()
    est = SklearnClassifier()

    scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

    scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)
    print(scores.mean(), scores.std())
