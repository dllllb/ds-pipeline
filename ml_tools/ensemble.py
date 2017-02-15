import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.externals.joblib import Parallel
from sklearn.base import clone as sk_clone


class ModelEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, assembly_estimator, intermediate_estimators, ensemble_train_size=.25, n_jobs=1):
        self.assembly_estimator = assembly_estimator
        self.intermediate_estimators = intermediate_estimators
        self.ensemble_train_size = ensemble_train_size
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.cross_validation import train_test_split

        if self.ensemble_train_size == 1:
            X_train1, y_train1 = X, y
            X_train2, y_train2 = X, y
        else:
            X_train1, X_train2, y_train1, y_train2 = train_test_split(X, y, test_size=self.ensemble_train_size)

        Parallel(n_jobs=self.n_jobs)(
            ((fit_est, [est, X_train1, y_train1], {}) for est in self.intermediate_estimators)
        )

        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X_train2], {}) for est in self.intermediate_estimators)
        ))

        self.assembly_estimator.fit(probas, y_train2)

        return self

    def predict(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X], {}) for est in self.intermediate_estimators)
        ))
        return self.assembly_estimator.predict(probas)

    def predict_proba(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X], {}) for est in self.intermediate_estimators)
        ))
        return self.assembly_estimator.predict_proba(probas)


def fit_est(estimator, features, labels):
    return estimator.fit(features, labels)


def predict_proba_est(estimator, features):
    return estimator.predict_proba(features)


class ModelEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 assembly_estimator,
                 intermediate_estimators,
                 ensemble_train_size=.25,
                 n_jobs=1):
        self.assembly_estimator = assembly_estimator
        self.intermediate_estimators = intermediate_estimators
        self.ensemble_train_size = ensemble_train_size
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.cross_validation import train_test_split

        if self.ensemble_train_size == 1:
            X_train, y_train = X, y
            X_holdout, y_holdout = X, y
        else:
            splits = train_test_split(X, y, train_size=self.ensemble_train_size)
            X_train, X_holdout, y_train, y_holdout = splits

        Parallel(n_jobs=self.n_jobs)(
            ((fit_est, [est, X_train, y_train], {}) for est in self.intermediate_estimators)
        )

        probas = np.vstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_est, [est, X_holdout], {}) for est in self.intermediate_estimators)
        )).T

        self.assembly_estimator.fit(probas, y_holdout)

        return self

    def predict(self, X):
        probas = np.vstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_est, [est, X], {}) for est in self.intermediate_estimators)
        )).T
        return self.assembly_estimator.predict(probas)


def predict_est(estimator, features):
    return estimator.predict(features)


class ModelEnsembleMean(BaseEstimator, ClassifierMixin):
    def __init__(self, intermediate_estimators, n_jobs=1):
        self.intermediate_estimators = intermediate_estimators
        self.n_jobs = n_jobs
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        Parallel(n_jobs=self.n_jobs)(
            ((fit_est, [est, X, y], {}) for est in self.intermediate_estimators)
        )

        return self

    def predict(self, X):
        rounded = self.predict_proba(X)

        if len(rounded.shape) == 1:
            indices = (rounded > 0).astype(np.int)
        else:
            indices = rounded.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        probas = np.dstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X], {}) for est in self.intermediate_estimators)
        ))

        rounded = np.mean(probas, axis=2)

        return rounded


class OneVsRestEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 assembly_estimator,
                 intermediate_estimators,
                 ensemble_train_size=.25,
                 n_jobs=1):
        self.assembly_estimator = assembly_estimator
        self.intermediate_estimators = intermediate_estimators
        self.ensemble_train_size = ensemble_train_size
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.cross_validation import train_test_split

        if self.ensemble_train_size == 1:
            X_train1, y_train1 = X, y
            X_train2, y_train2 = X, y
        else:
            splits = train_test_split(X, y, test_size=self.ensemble_train_size)
            X_train1, X_train2, y_train1, y_train2 = splits

        from sklearn.preprocessing import LabelBinarizer
        labels = LabelBinarizer().fit_transform(y_train1).T

        Parallel(n_jobs=self.n_jobs)(
            (
                (fit_est, [est, X_train1, labels_bin], {})
                for est, labels_bin in zip(self.intermediate_estimators, labels)
            )
        )

        probas = self.intermediate_predict_proba(X_train2)

        self.assembly_estimator.fit(probas, y_train2)

        return self

    def intermediate_predict_proba(self, X):
        def predict_proba_est_bin(estimator, features):
            return estimator.predict_proba(features).T[1]

        probas = np.array(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est_bin, [est, X], {}) for est in self.intermediate_estimators)
        )).T
        return probas

    def predict(self, X):
        probas = self.intermediate_predict_proba(X)
        return self.assembly_estimator.predict(probas)

    def predict_proba(self, X):
        probas = self.intermediate_predict_proba(X)
        return self.assembly_estimator.predict_proba(probas)


class ModelEnsembleMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, intermediate_estimators, n_jobs=1):
        self.intermediate_estimators = intermediate_estimators
        self.n_jobs = n_jobs

    def fit(self, X, y):
        Parallel(n_jobs=self.n_jobs)(
            ((fit_est, [est, X, y], {}) for est in self.intermediate_estimators)
        )

        return self

    def predict(self, X):
        preds = Parallel(n_jobs=self.n_jobs)(
            ((predict_est, [est, X], {}) for est in self.intermediate_estimators)
        )

        rounded = np.mean(np.array(preds).T, axis=1)

        return rounded


class KFoldStackingFullRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, final_estimator, intermediate_estimators, n_jobs=1, n_folds=3):
        self.final_estimator = final_estimator
        self.intermediate_estimators = intermediate_estimators
        self.n_jobs = n_jobs
        self.n_folds = n_folds

        self.final_est = None
        self.intermediate_ests = None

    def fit(self, x, y):
        from sklearn.cross_validation import KFold

        folds = KFold(n=len(y), n_folds=self.n_folds, shuffle=True)

        train_folds, test_folds = zip(*folds)

        intermediate_ests = Parallel(n_jobs=self.n_jobs)(
            (
                (kfold_fit, [est, x, y, train_folds], {})
                for est in self.intermediate_estimators
            )
        )

        probas = np.vstack(Parallel(n_jobs=self.n_jobs)(
            (
                (kfold_predict_est, [est, x, test_folds], {})
                for est in intermediate_ests
            )
        )).T

        y_test = np.hstack([y[idx] for idx in test_folds])

        self.final_est = sk_clone(self.final_estimator).fit(probas, y_test)

        self.intermediate_ests = Parallel(n_jobs=self.n_jobs)(
            ((fit_est_clone, [est, x, y], {}) for est in self.intermediate_estimators)
        )

        return self

    def predict(self, x):
        probas = np.vstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_est, [est, x], {}) for est in self.intermediate_ests)
        )).T

        return self.final_est.predict(probas)


def fit_est_clone(estimator, features, labels):
    return sk_clone(estimator).fit(features, labels)


def kfold_fit(estimator, X, y, folds):
    ests = [
        sk_clone(estimator).fit(X[idx], y[idx])
        for idx in folds
    ]
    return ests


def kfold_predict_est(estimators, x, folds):
    probas = np.hstack([est.predict(x[idx]) for est, idx in zip(estimators, folds)])
    return probas


class KFoldStackingFull(BaseEstimator, ClassifierMixin):
    def __init__(self, final_estimator, intermediate_estimators, n_jobs=1, n_folds=3):
        self.final_estimator = final_estimator
        self.intermediate_estimators = intermediate_estimators
        self.n_jobs = n_jobs
        self.n_folds = n_folds

        self.final_est = None
        self.intermediate_ests = None

    def fit(self, X, y):
        from sklearn.cross_validation import StratifiedKFold

        folds = StratifiedKFold(y, n_folds=self.n_folds, shuffle=True)

        train_folds, test_folds = zip(*folds)

        intermediate_ests = Parallel(n_jobs=self.n_jobs)(
            (
                (kfold_fit, [est, X, y, train_folds], {})
                for est in self.intermediate_estimators
            )
        )

        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            (
                (kfold_predict_proba_est, [est, X, test_folds], {})
                for est in intermediate_ests
            )
        ))

        y_test = np.hstack([y[idx] for idx in test_folds])

        self.final_est = sk_clone(self.final_estimator).fit(probas, y_test)

        self.intermediate_ests = Parallel(n_jobs=self.n_jobs)(
            ((fit_est_clone, [est, X, y], {}) for est in self.intermediate_estimators)
        )

        return self

    def predict(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X], {}) for est in self.intermediate_ests)
        ))

        return self.final_est.predict(probas)

    def predict_proba(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X], {}) for est in self.intermediate_ests)
        ))

        return self.final_est.predict_proba(probas)


def kfold_predict_proba_est(estimators, x, folds):
    pr = [est.predict_proba(x[idx]) for est, idx in zip(estimators, folds)]
    probas = np.vstack(pr)
    return probas


class KFoldStacking(BaseEstimator, ClassifierMixin):
    def __init__(self, final_estimator, intermediate_estimators, n_jobs=1, n_folds=3):
        self.final_estimator = final_estimator
        self.intermediate_estimators = intermediate_estimators
        self.n_jobs = n_jobs
        self.n_folds = n_folds

        self.final_est = None
        self.intermediate_ests = None

    def fit(self, X, y):
        from sklearn.cross_validation import StratifiedKFold

        folds = StratifiedKFold(y, n_folds=self.n_folds, shuffle=True)

        train_folds, test_folds = zip(*folds)

        self.intermediate_ests = Parallel(n_jobs=self.n_jobs)(
            (
                (kfold_fit, [est, X, y, train_folds], {})
                for est in self.intermediate_estimators
            )
        )

        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            (
                (kfold_predict_proba_est, [est, X, test_folds], {})
                for est in self.intermediate_ests
            )
        ))

        y_test = np.hstack([y[idx] for idx in test_folds])

        self.final_est = sk_clone(self.final_estimator).fit(probas, y_test)

        return self

    def predict(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((kfold_predict_proba_mean, [ests, X], {}) for ests in self.intermediate_ests)
        ))
        return self.final_est.predict(probas)

    def predict_proba(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((kfold_predict_proba_mean, [ests, X], {}) for ests in self.intermediate_ests)
        ))
        return self.final_est.predict_proba(probas)


def kfold_predict_proba_mean(estimators, features):
    probas = np.dstack([est.predict_proba(features) for est in estimators])
    proba_means = np.mean(probas, axis=2)
    return proba_means


class ForcedMultilabelModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.estimators = None

    def fit(self, X, y):
        from sklearn.preprocessing import LabelBinarizer

        labels_bin = LabelBinarizer().fit_transform(y)

        self.estimators = Parallel(n_jobs=self.n_jobs)(
            ((fit_est_clone, [self.estimator, X, labels], {}) for labels in labels_bin.T)
        )

        return self

    def predict(self, X):
        return np.array([estimator.predict(X) for estimator in self.estimators]).T

    def predict_proba(self, X):
        return np.array([est.predict_proba(X)[:, 1] for est in self.estimators]).T


class PerGroupRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, split_condition, n_jobs=1, verbose=0):
        self.split_condition = split_condition
        self.estimator = estimator
        self.group_estimators = None
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        self.group_estimators = dict(Parallel(n_jobs=self.n_jobs)(
            (
                (fit_clone_with_key, [self.estimator, gdf, y.ix[gdf.index], gkey], {})
                for gkey, gdf in X.groupby(self.split_condition)
            )
        ))

        return self

    def predict(self, X):
        preds = pd.concat(Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            (
                (predict_with_id, [self.group_estimators[gkey], gdf], {})
                for gkey, gdf in X.groupby(self.split_condition)
            )
        ))

        return preds


def predict_with_id(est, fdf):
    return pd.Series(est.predict(fdf), index=fdf.index)


def fit_clone_with_key(estimator, features, labels, key):
    return key, sk_clone(estimator).fit(features, labels)
