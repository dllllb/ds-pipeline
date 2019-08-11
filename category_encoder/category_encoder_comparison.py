import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

from dstools.ml.categorical import empirical_bayes_encoder, multi_class_empirical_bayes_encoder
from dstools.ml.categorical import count_encoder
from dstools.ml.categorical import empirical_bayes_vibrant_encoder, mc_empirical_bayes_vibrant_encoder
from dstools.ml.categorical import yandex_mean_encoder, mc_yandex_mean_encoder
from dstools.ml.categorical import noisy_mean_encoder, mc_noisy_mean_encoder
from dstools.ml.categorical import kfold_target_mean_encoder
from dstools.ml.categorical import mc_kfold_target_mean_encoder


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
    else:
        raise AssertionError(f'unkonwn category encoding: {category_encoding}')

    transf = make_pipeline(transf, SimpleImputer(strategy=params['imputation']))

    est_type = params['est_type']

    if est_type == 'rf':
        est = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_features=params['max_features'],
            max_depth=params['max_depth'],
            random_state=1)
    else:
        raise AssertionError(f'unkonwn estimator: {est_type}')

    return make_pipeline(transf, est)


def titanic_dataset(_):
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


def beeline_dataset(_):
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


def update_model_stats(stats_file, params, results):
    import json
    import os.path

    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []

    stats.append({**results, **params})

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)


def run_experiment(est, dataset, scorer, params):
    import time

    start = time.time()
    if params['valid_type'] == 'cv':
        cv = params['n_folds']
        features, target = dataset(params)
        scores = cv_test(est(params), features, target, scorer, cv)
    exec_time = time.time() - start
    return {**scores, 'exec-time-sec': exec_time}


def cv_test(est, features, target, scorer, cv):
    scores = cross_val_score(est, features, target, scoring=scorer, cv=cv)
    return {'score-mean': scores.mean(), 'score-std': scores.std()}
