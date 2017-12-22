from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_boston, load_iris
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing.label import label_binarize
import pandas as pd

from dstools.ml.xgboost_tools import XGBoostRegressor, XGBoostClassifier


def roc_auc_avg_score(y_true, y_score):
    y_bin = label_binarize(y_true, classes=sorted(set(y_true)))
    return roc_auc_score(y_bin, y_score)


def test_regressor():
    boston = load_boston()
    est = XGBoostRegressor(num_rounds=50, objective='reg:linear', silent=1)

    scores = cross_val_score(estimator=est, X=boston.data, y=boston.target, cv=3)

    print(scores.mean(), scores.std())


def test_classifier():
    iris = load_iris()
    est = XGBoostClassifier(num_rounds=50, objective='multi:softprob', num_class=3, silent=1)

    scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

    scores = cross_val_score(estimator=est, X=iris.data, y=iris.target, cv=3, scoring=scorer)

    print(scores.mean(), scores.std())


def test_classifier_shifted_labels():
    iris = load_iris()
    est = XGBoostClassifier(num_rounds=50, objective='multi:softprob', num_class=3, silent=1)

    scorer = make_scorer(roc_auc_avg_score, needs_proba=True)

    scores = cross_val_score(estimator=est, X=iris.data, y=iris.target+1, cv=3, scoring=scorer)

    print(scores.mean(), scores.std())


def test_classifier_bin():
    iris = load_iris()
    target_bin = LabelBinarizer().fit_transform(iris.target).T[0]
    est = XGBoostClassifier(num_rounds=50, objective='reg:logistic', silent=1)

    scores = cross_val_score(estimator=est, X=iris.data, y=target_bin, cv=3, scoring='roc_auc')

    print(scores.mean(), scores.std())


def test_feature_importance():
    iris = load_iris()
    est = XGBoostClassifier(num_rounds=50, objective='multi:softprob', num_class=3, silent=1)
    est.fit(pd.DataFrame(iris.data, columns=iris.feature_names), iris.target)

    print(est.get_fscore())


def test_classifier_bin_predict():
    iris = load_iris()
    target_bin = LabelBinarizer().fit_transform(iris.target).T[0]
    est = XGBoostClassifier(num_rounds=50, objective='binary:logistic', silent=1)

    split = train_test_split(iris.data, target_bin)
    x_train, x_test, y_train, y_test = split

    est.fit(x_train, y_train)

    preds = est.predict(x_test)

    print(zip(preds, y_test))


def test_pickle():
    est = XGBoostClassifier(num_rounds=50, objective='binary:logistic', silent=1)
    from sklearn.base import clone as sk_clone
    cl = sk_clone(est)
    assert(cl.params['num_rounds'] == 50)
