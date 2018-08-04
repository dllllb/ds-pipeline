import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Imputer
from sklearn.ensemble import RandomForestClassifier

from dstools.ml.experiment import run_experiment
from dstools.ml.transformers import empirical_bayes_encoder, count_encoder, empirical_bayes_vibrant_encoder


def default_estimator(params):
    category_encoding = params['category_encoding']

    if category_encoding == 'onehot':
        df2dict = FunctionTransformer(
            lambda x: x.to_dict(orient='records'), validate=False)
        transf = make_pipeline(
            df2dict,
            DictVectorizer(sparse=False),
        )
    elif category_encoding == 'empyrical_bayes':
        transf = empirical_bayes_encoder()
    elif category_encoding == 'empyrical_bayes_vibrant':
        transf = empirical_bayes_vibrant_encoder(prior_est_frac=.3)
    elif category_encoding == 'count':
        transf = count_encoder()

    transf = make_pipeline(transf, Imputer(strategy='most_frequent'))

    est_type = params['est_type']

    if est_type == 'rfr':
        est = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_features=params['max_features'],
            max_depth=params['max_depth'],
            random_state=1)

    return make_pipeline(transf, est)


def titanic_dataet(params):
    import os
    df = pd.read_csv(f'{os.path.dirname(__file__)}/titanic.csv')
    df = df.replace(r'\s+', np.nan, regex=True)
    features = df.drop(['survived', 'alive'], axis=1)
    return features, df.survived


def titanic_experiment(overrides):
    titanic_params = {
        'est_type': 'rfr',
        'valid_type': 'cv',
        'n_folds': 5,
        'n_jobs': 1,
        'n_estimators': 50,
        'max_depth': 4,
        'max_features': 'auto'
    }
    params = {**titanic_params, **overrides}
    run_experiment(default_estimator, titanic_dataet, 'roc_auc', params, 'titanic.json')
