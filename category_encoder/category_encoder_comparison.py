import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Imputer
from sklearn.ensemble import RandomForestClassifier

from dstools.ml.experiment import run_experiment
from dstools.ml.transformers import empirical_bayes_encoder, multi_class_empirical_bayes_encoder
from dstools.ml.transformers import count_encoder
from dstools.ml.transformers import empirical_bayes_vibrant_encoder, mc_empirical_bayes_vibrant_encoder
from dstools.ml.transformers import yandex_mean_encoder, mc_yandex_mean_encoder
from dstools.ml.transformers import noisy_mean_encoder, mc_noisy_mean_encoder
from dstools.ml.transformers import kfold_target_mean_encoder
from dstools.ml.transformers import mc_kfold_target_mean_encoder


def default_estimator(params):
    category_encoding = params['category_encoding']
    multi_class = params['multi_class']

    if category_encoding == 'onehot':
        df2dict = FunctionTransformer(
            lambda x: x.to_dict(orient='records'), validate=False)
        transf = make_pipeline(
            df2dict,
            DictVectorizer(sparse=False),
        )
    elif category_encoding == 'empirical_bayes':
        if multi_class:
            transf = multi_class_empirical_bayes_encoder()
        else:
            transf = empirical_bayes_encoder()
    elif category_encoding == 'empirical_bayes_vibrant':
        if multi_class:
            transf = mc_empirical_bayes_vibrant_encoder(prior_est_frac=.3)
        else:
            transf = empirical_bayes_vibrant_encoder(prior_est_frac=.3)
    elif category_encoding == 'count':
        transf = count_encoder()
    elif category_encoding == 'yandex_mean':
        if multi_class:
            transf = mc_yandex_mean_encoder()
        else:
            transf = yandex_mean_encoder()
    elif category_encoding == 'noisy_mean':
        if multi_class:
            transf = mc_noisy_mean_encoder()
        else:
            transf = noisy_mean_encoder()
    elif category_encoding == 'kfold_mean':
        if multi_class:
            transf = mc_kfold_target_mean_encoder()
        else:
            transf = kfold_target_mean_encoder()


    transf = make_pipeline(transf, Imputer(strategy=params['imputation']))

    est_type = params['est_type']

    if est_type == 'rf':
        est = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_features=params['max_features'],
            max_depth=params['max_depth'],
            random_state=1)

    return make_pipeline(transf, est)


def titanic_dataset(params):
    import os
    df = pd.read_csv(f'{os.path.dirname(__file__)}/titanic.csv')
    df = df.replace(r'\s+', np.nan, regex=True)
    features = df.drop(['survived', 'alive'], axis=1)
    return features, df.survived


def titanic_experiment(overrides):
    titanic_params = {
        'est_type': 'rf',
        'valid_type': 'cv',
        'n_folds': 5,
        'n_jobs': 1,
        'n_estimators': 50,
        'max_depth': 4,
        'max_features': 'auto',
        'imputation': 'most_frequent',
        'multi_class': False,
    }
    params = {**titanic_params, **overrides}
    results = run_experiment(default_estimator, titanic_dataset, 'roc_auc', params, 'titanic.json')
    update_model_stats('titanic.json', params, results)


def beeline_dataset(params):
    df = pd.read_csv(f'{os.path.dirname(__file__)}/beeline-ss20.csv.gz')
    features = df.drop('y', axis=1)
    return features, df.y


def beeline_experiment(overrides):
    titanic_params = {
        'est_type': 'rf',
        'valid_type': 'cv',
        'n_folds': 5,
        'n_jobs': 1,
        'n_estimators': 50,
        'max_depth': 4,
        'max_features': 'auto',
        'imputation': 'most_frequent',
        'multi_class': True,
    }
    params = {**titanic_params, **overrides}
    results = run_experiment(default_estimator, beeline_dataset, 'accuracy', params)
    update_model_stats('beeline.json', params, results)
