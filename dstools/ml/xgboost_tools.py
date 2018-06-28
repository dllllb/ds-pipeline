"""
XGBoost scikit-learn wrapper
Advantages over default wrapper:
- it is possible to setup validation holdout and early stopping during estimator configuration
  - hence it is possible to use holdout and early stopping in pipelines
  - holdout is passed as additional parameter to the fit method in default wrapper
"""

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np


class XGBoostModel(BaseEstimator):
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, x, y):
        n_rounds = self.params.get('num_rounds', 100)
        n_es_rounds = self.params.get('num_es_rounds', 50)
        early_stop_share = self.params.get('es_share', 0.05)
        feval = self.params.get('eval_func')
        maximize_eval = self.params.get('maximize_eval', False)
        obj = self.params.get('objective_func')
        missing_marker = self.params.get('missing_marker', np.NaN)
        ybin_func = self.params.get('ybin_func')
        objective = self.params.get('objective')
        verbose = self.params.get('verbose', n_es_rounds)

        if early_stop_share > 0:
            if ybin_func:
                split = train_test_split(x, y, test_size=early_stop_share, stratify=ybin_func(y))
            else:
                if objective in ['multi:softprob', 'binary:logistic', 'multi:softmax']:
                    split = train_test_split(x, y, test_size=early_stop_share, stratify=y)
                else:
                    split = train_test_split(x, y, test_size=early_stop_share)
            x_train, x_es, y_train, y_es = split
            dm = xgb.DMatrix(x_train, label=y_train, missing=missing_marker)
            es = xgb.DMatrix(x_es, label=y_es, missing=missing_marker)
            self.model = xgb.train(
                params=self.params,
                dtrain=dm,
                num_boost_round=n_rounds,
                evals=[(dm, 'train'), (es, 'validation')],
                verbose_eval=verbose,
                early_stopping_rounds=n_es_rounds,
                feval=feval,
                maximize=maximize_eval,
                obj=obj,
            )
        else:
            dm = xgb.DMatrix(x, label=y, missing=missing_marker)
            self.model = xgb.train(self.params, dm, n_rounds, feval=feval)

        return self

    def xgb_predict(self, x):
        if not self.model:
            raise AttributeError('model is not created, call fit method first')

        booster = self.params.get('booster', 'gbtree')
        best_iteration = getattr(self.model, 'best_ntree_limit', None)

        missing_marker = self.params.get('missing_marker', np.NaN)

        if best_iteration:
            if booster == 'gblinear':
                pred = self.model.predict(xgb.DMatrix(x, missing=missing_marker))
            else:
                pred = self.model.predict(xgb.DMatrix(x, missing=missing_marker), ntree_limit=best_iteration)
        else:
            pred = self.model.predict(xgb.DMatrix(x, missing=missing_marker))

        return pred

    # __init__ key-value args are not considered in sklearn.base.BaseEstimator.get_params
    # hence explicit get_params is required
    def get_params(self, deep=True):
        return self.params

    def get_fscore(self):
        return self.model.get_fscore()

    @property
    def feature_importances_(self):
        return self.model.get_fscore().values()


class XGBoostRegressor(XGBoostModel, RegressorMixin):
    def __init__(self, **params):
        super(XGBoostRegressor, self).__init__(**params)

    def predict(self, x):
        return self.xgb_predict(x)


class XGBoostClassifier(XGBoostModel, ClassifierMixin):
    def __init__(self, **params):
        super(XGBoostClassifier, self).__init__(**params)
        self.le = LabelEncoder()

    def fit(self, x, y):
        y_xgb = self.le.fit_transform(y)
        super(XGBoostClassifier, self).fit(x, y_xgb)
        return self

    def predict(self, x):
        probas = self.predict_proba(x)

        if self.params.get('objective') == 'multi:softprob':
            idx = np.argmax(probas, axis=1)
            return self.le.inverse_transform(idx)
        else:
            return self.le.inverse_transform((probas[:, 1] > .5).astype(np.int8))

    def predict_proba(self, x):
        pred = self.xgb_predict(x)

        if self.params.get('objective') == 'multi:softprob':
            return pred
        else:
            classone_probs = pred
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()
