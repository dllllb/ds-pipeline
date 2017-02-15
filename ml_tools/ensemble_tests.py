#!/usr/bin/env python
# coding: utf-8

import unittest
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.datasets import load_iris, load_boston
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeRegressor

from ensemble import ModelEnsemble, ModelEnsembleRegressor, ModelEnsembleMean, OneVsRestEnsemble, \
    ModelEnsembleMeanRegressor, KFoldStackingFullRegressor, KFoldStackingFull, KFoldStacking, ForcedMultilabelModel, \
    PerGroupRegressor


def roc_auc_avg_score(y_true, y_score):
    y_bin = label_binarize(y_true, classes=sorted(set(y_true)))
    return roc_auc_score(y_bin, y_score)


class TestModelEnsemble(unittest.TestCase):
    def test_simple(self):
        iris = load_iris()

        est = ModelEnsemble(
            assembly_estimator=RandomForestClassifier(n_estimators=50, random_state=1),
            intermediate_estimators=[
                LogisticRegression(random_state=1),
                GaussianNB()
            ]
        )

        scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

        scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)

        print(scores.mean(), scores.std())


class TestModelEnsembleRegressor(unittest.TestCase):
    def test_simple(self):
        boston = load_boston()
        est = ModelEnsembleRegressor(
            intermediate_estimators=[
                LinearRegression(),
                RandomForestRegressor(n_estimators=5)
            ],
            assembly_estimator=LinearRegression()
        )

        scores = cross_val_score(estimator=est, X=boston.data, y=boston.target, cv=3)

        print(scores.mean(), scores.std())


class TestModelEnsembleMean(unittest.TestCase):
    def test_simple(self):
        iris = load_iris()
        est = ModelEnsembleMean(
            intermediate_estimators=[
                LogisticRegression(random_state=1),
                GaussianNB()
            ]
        )
        scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

        scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)

        print(scores.mean(), scores.std())


class TestOneVsRestEnsemble(unittest.TestCase):
    def test_simple(self):
        iris = load_iris()
        est = OneVsRestEnsemble(
            intermediate_estimators=[
                LogisticRegression(),
                GaussianNB(),
                RandomForestClassifier(n_estimators=5)
            ],
            assembly_estimator=RandomForestClassifier(n_estimators=10)
        )

        scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

        scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)

        print(scores.mean(), scores.std())


class TestModelEnsembleMeanRegressor(unittest.TestCase):
    def test_simple(self):
        boston = load_boston()
        est = ModelEnsembleMeanRegressor(
            intermediate_estimators=[
                LinearRegression(),
                RandomForestRegressor(n_estimators=5)
            ]
        )

        scores = cross_val_score(estimator=est, X=boston.data, y=boston.target, cv=3)

        print(scores.mean(), scores.std())


class TestKFoldStackingFullRegressor(unittest.TestCase):
    def test_simple(self):
        boston = load_boston()
        est = KFoldStackingFullRegressor(
            final_estimator=RandomForestRegressor(n_estimators=50, random_state=1),
            intermediate_estimators=[
                Ridge(random_state=1),
                DecisionTreeRegressor()
            ],
            n_folds=2
        )

        scores = cross_val_score(estimator=est, X=boston.data, y=boston.target, cv=3)

        print(scores.mean(), scores.std())


class TestKFoldStackingFull(unittest.TestCase):
    def test_simple(self):
        iris = load_iris()
        est = KFoldStackingFull(
            final_estimator=LogisticRegression(),
            intermediate_estimators=[
                LogisticRegression(random_state=1),
                GaussianNB()
            ],
            n_folds=2
        )

        scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

        scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)

        print(scores.mean(), scores.std())


class TestKFoldStacking(unittest.TestCase):
    def test_simple(self):
        iris = load_iris()
        est = KFoldStacking(
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=1),
            intermediate_estimators=[
                LogisticRegression(random_state=1),
                GaussianNB()
            ],
            n_folds=2
        )

        scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

        scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)

        print(scores.mean(), scores.std())


class TestForcedMultilabelModel(unittest.TestCase):
    def test_simple(self):
        iris = load_iris()
        est = ForcedMultilabelModel(LogisticRegression())

        scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

        scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)

        print(scores.mean(), scores.std())


class TestPerGroupRegressor(unittest.TestCase):
    def test_simple(self):
        boston = load_boston()
        features = pd.DataFrame(boston.data, columns=boston.feature_names)
        features['group'] = (features.AGE % 3).astype(int)
        labels = pd.Series(boston.target)
        est = PerGroupRegressor(
            estimator=LinearRegression(),
            split_condition='group',
        )

        scores = cross_val_score(estimator=est, X=features, y=labels, cv=3)

        print(scores.mean(), scores.std())


if __name__ == '__main__':
    unittest.main()
