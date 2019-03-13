from sklearn.model_selection import cross_val_score


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
