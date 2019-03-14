import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing.label import label_binarize


def roc_auc_avg_score(y_true, y_score):
    y_bin = label_binarize(y_true, classes=sorted(set(y_true)))
    return roc_auc_score(y_bin, y_score)


def test_h2o_skearn():
    pytest.importorskip('h2o')

    from dstools.ml.h2o import H2ODecorator

    iris = load_iris()
    est = H2ODecorator('glm', {})

    scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

    scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)
    print(scores.mean(), scores.std())
