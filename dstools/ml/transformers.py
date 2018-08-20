from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
import pandas as pd
import six
from functools import partial


def df2dict():
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(
        lambda x: x.to_dict(orient='records'), validate=False)


def constant_value_imputer(value):
    return FunctionTransformer(lambda df: df.fillna(value), validate=False)


def map_encoder(vals, mapping):
    return vals.map(lambda x: mapping.get(x, mapping.get('nan', 0)))


def beta_encoder(vals, mapping):
    res = vals.replace(np.nan, 'nan').copy()
    input_cats = res.value_counts()
    for cat, cnt in input_cats.items():
        if cat in mapping:
            alpha, beta = mapping.get(cat)
        else:
            alpha, beta = mapping.get('nan')
        res[res == cat] = np.random.beta(alpha, beta, cnt)
    return res


def kfold_category_encoder(df, y, builder, columns=None, n_jobs=1, true_label=None, n_folds=3):
    kf = KFold(n_folds)
    kf


class TargetCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, builder, columns=None, n_jobs=1, true_label=None):
        self.vc = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.true_label = true_label
        self.builder = builder

    def fit(self, df, y=None):
        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        if self.true_label is not None:
            target = (y == self.true_label)
        else:
            target = y

        encoders = Parallel(n_jobs=self.n_jobs)(
            delayed(self.builder)(df[col], target)
            for col in columns
        )

        self.vc = dict(zip(columns, encoders))

        return self

    def transform(self, df):
        res = df.copy()
        for col, encoder in self.vc.items():
            res[col] = encoder(res[col])
        return res


def kfold_encoder(builder, col, kf, target):
    res = col.copy()
    for train_idx, test_idx in kf.split(col):
        encoder = builder(col[train_idx], target)
        res[test_idx] = encoder(res[test_idx])
    return res


class KFoldTargetCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, builder, columns=None, n_jobs=1, true_label=None, n_folds=3):
        self.vc = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.true_label = true_label
        self.builder = builder
        self.n_folds = n_folds

    def fit(self, df, y=None):
        return self

    def fit_transform(self, df, y=None, **kwargs):
        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        if self.true_label is not None:
            target = (y == self.true_label)
        else:
            target = y

        kf = KFold(self.n_folds)

        encoded_cols = Parallel(n_jobs=self.n_jobs)(
            delayed(
                kfold_encoder
            )(self.builder, df[col], kf, target)
            for col in columns
        )

        res = df.copy()
        for col, vals in zip(columns, encoded_cols):
            res[col] = vals

        return res


def build_zeroing_encoder(column, __, threshold, top, placeholder):
    vc = column.replace(np.nan, 'nan').value_counts()
    candidates = set(vc[vc <= threshold].index).union(set(vc[top:].index))
    mapping = dict(zip(vc.index, vc.index))
    if 'nan' in mapping:
        mapping['nan'] = np.nan
    for c in candidates:
        mapping[c] = placeholder
    return partial(map_encoder, mapping=mapping)


def high_cardinality_zeroing(threshold=1, top=10000, placeholder='zeroed', columns=None, n_jobs=1):
    buider = partial(
        build_zeroing_encoder,
        threshold=threshold,
        top=top,
        placeholder=placeholder
    )

    return TargetCategoryEncoder(buider, columns, n_jobs)


def build_count_encoder(column, __):
    entries = column.replace(np.nan, 'nan').value_counts()
    entries = entries.sort_values(ascending=False).index
    mapping = dict(zip(entries, range(len(entries))))
    return partial(map_encoder, mapping=mapping)


def count_encoder(columns=None, n_jobs=1):
    return TargetCategoryEncoder(build_count_encoder, columns, n_jobs)


def build_categorical_feature_encoder_mean(column, target, size_threshold):
    global_mean = target.mean()
    col_dna = column.fillna('nan')
    means = target.groupby(col_dna).mean()
    counts = col_dna.groupby(col_dna).count()
    reg = pd.DataFrame(counts / (counts + size_threshold))
    reg[1] = 1.
    reg = reg.min(axis=1)
    means_reg = means * reg + (1-reg) * global_mean
    entries = means_reg.sort_values(ascending=False).index

    mapping = dict(zip(entries, range(len(entries))))
    return partial(map_encoder, mapping=mapping)


def target_mean_encoder(columns=None, n_jobs=1, size_threshold=10, true_label=None):
    buider = partial(
        build_categorical_feature_encoder_mean,
        size_threshold=size_threshold
    )
    return TargetCategoryEncoder(buider, columns, n_jobs, true_label)


def kfold_target_mean_encoder(columns=None, n_jobs=1, size_threshold=10, true_label=None, n_folds=3):
    buider = partial(
        build_categorical_feature_encoder_mean,
        size_threshold=size_threshold
    )
    return KFoldTargetCategoryEncoder(buider, columns, n_jobs, true_label, n_folds)


def multi_class_target_share_encoder(columns=None, n_jobs=1, size_threshold=10):
    builder = partial(
        build_categorical_feature_encoder_mean,
        size_threshold=size_threshold
    )
    return MultiClassTargetCategoryEncoder(builder, columns, n_jobs)


def build_yandex_mean_encoder(column, target, alpha):
    global_target_mean = target.mean()
    col_dna = column.fillna('nan')
    cat_pos = target.groupby(col_dna).sum()
    cat_count = col_dna.groupby(col_dna).count()

    codes = (alpha * global_target_mean + cat_pos) / (alpha + cat_count)

    return partial(map_encoder, mapping=codes.to_dict())


def yandex_mean_encoder(columns=None, n_jobs=1, alpha=100, true_label=None):

    """
    Smoothed mean-encoding with custom smoothing strength (alpha)
    http://learningsys.org/nips17/assets/papers/paper_11.pdf
    """

    buider = partial(
        build_yandex_mean_encoder,
        alpha=alpha,
    )
    return TargetCategoryEncoder(buider, columns, n_jobs, true_label)


def mc_yandex_mean_encoder(columns=None, n_jobs=1, alpha=100):
    buider = partial(
        build_yandex_mean_encoder,
        alpha=alpha,
    )
    return MultiClassTargetCategoryEncoder(buider, columns, n_jobs)


def build_noisy_mean_encoder(column, target, alpha):
    noise = np.random.uniform(size=column.shape)
    col_dna = column.fillna('nan')
    cat_pos = target.groupby(col_dna).sum()
    noise = cat_pos.copy()
    noise.loc[:] = np.random.uniform(size=noise.shape)    
    cat_count = col_dna.groupby(col_dna).count()

    codes = (alpha * noise + cat_pos) / (alpha + cat_count)

    return partial(map_encoder, mapping=codes.to_dict())


def noisy_mean_encoder(columns=None, n_jobs=1, alpha=100, seed=0, true_label=None):

    """
    Mean-encoding smoothed with noise
    Avoids overfitting on features with high cardinality
    """

    np.random.seed(seed)
    buider = partial(
        build_noisy_mean_encoder,
        alpha=alpha,
    )
    return TargetCategoryEncoder(buider, columns, n_jobs, true_label)


def mc_noisy_mean_encoder(columns=None, n_jobs=1, alpha=100, seed=0):
    np.random.seed(seed)
    buider = partial(
        build_noisy_mean_encoder,
        alpha=alpha,
    )
    return MultiClassTargetCategoryEncoder(buider, columns, n_jobs)


def build_empirical_bayes_encoder(column, target, prior_est_frac=.1):
    if prior_est_frac < .999:
        taret_subsample = target.subsample(prior_est_frac)
    else:
        taret_subsample = target

    global_pos = taret_subsample.sum()
    global_count = taret_subsample.count()
    col_dna = column.fillna('nan')
    cat_pos = target.groupby(col_dna).sum()
    cat_count = col_dna.groupby(col_dna).count()

    codes = (global_pos + cat_pos) / (global_count + cat_count)

    return partial(map_encoder, mapping=codes.to_dict())


def empirical_bayes_encoder(columns=None, n_jobs=1, true_label=None, prior_est_frac=1):
    builder = partial(build_empirical_bayes_encoder, prior_est_frac=prior_est_frac)
    return TargetCategoryEncoder(builder, columns, n_jobs, true_label)


def multi_class_empirical_bayes_encoder(columns=None, n_jobs=1, prior_est_frac=1):
    builder = partial(build_empirical_bayes_encoder, prior_est_frac=prior_est_frac)
    return MultiClassTargetCategoryEncoder(builder, columns, n_jobs)


def build_empirical_bayes_vibrant_encoder(column, target, prior_est_frac=1):
    if prior_est_frac < .999:
        sample = np.random.randint(0, len(target), int(len(target)*prior_est_frac))
        taret_subsample = target.iloc[sample]
    else:
        taret_subsample = target

    global_pos = taret_subsample.sum()
    global_count = taret_subsample.count()
    col_dna = column.fillna('nan')
    cat_pos = target.groupby(col_dna).sum()
    cat_count = col_dna.groupby(col_dna).count()

    alpha = global_pos + cat_pos
    beta = global_count - global_pos + cat_count - cat_pos

    codes = dict((k, tuple(v)) for k, v in pd.DataFrame([alpha, beta]).items())

    if 'nan' not in codes:
        codes['nan'] = (global_pos, global_count-global_pos)

    return partial(beta_encoder, mapping=codes)


def empirical_bayes_vibrant_encoder(columns=None, n_jobs=1, true_label=None, prior_est_frac=1):
    builder = partial(build_empirical_bayes_vibrant_encoder, prior_est_frac=prior_est_frac)
    return TargetCategoryEncoder(builder, columns, n_jobs, true_label)


def mc_empirical_bayes_vibrant_encoder(columns=None, n_jobs=1, prior_est_frac=1):
    builder = partial(build_empirical_bayes_vibrant_encoder, prior_est_frac=prior_est_frac)
    return MultiClassTargetCategoryEncoder(builder, columns, n_jobs)


def build_empirical_bayes_encoder_normal(column, target):
    # https://stats.stackexchange.com/questions/237037/bayesian-updating-with-new-data
    global_mean, global_var = target.mean(), target.var()
    col_dna = column.fillna('nan')
    cat_mean = target.groupby(col_dna).sum()
    cat_var = target.groupby(col_dna).var()

    codes = (cat_mean*cat_var + global_mean*global_var) / (global_var + cat_var)

    return partial(map_encoder, mapping=codes.to_dict())


def empirical_bayes_encoder_normal(columns=None, n_jobs=1):
    builder = build_empirical_bayes_encoder_normal
    return TargetCategoryEncoder(builder, columns, n_jobs, true_label=None)


class MultiClassTargetCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, builder, columns=None, n_jobs=1):
        self.class_encodings = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.builder = builder

    def fit(self, df, y=None):
        encoded_classes = pd.Series(y).value_counts().index[1:]

        if self.columns is None:
            self.columns = df.select_dtypes(include=['object'])

        for cl in encoded_classes:
            vc = dict(zip(self.columns, Parallel(n_jobs=self.n_jobs)(
                delayed(self.builder)(df[col], pd.Series(y == cl))
                for col in self.columns
            )))
            self.class_encodings[cl] = vc

        return self

    def transform(self, df):
        res = df.copy()
        for cls, cols in self.class_encodings.items():
            for col, encoder in cols.items():
                res['{}_{}'.format(col, cls)] = encoder(res[col])

        res = res.drop(self.columns, axis=1)
        return res


def field_list_func(df, field_names, drop_mode=False, ignore_case=True):
    if ignore_case:
        field_names = list(map(six.u, field_names))
        field_names = list(map(lambda e: e.lower(), field_names))

        df_cols = list(map(six.u, df.columns))
        df_cols = list(map(lambda e: e.lower(), df_cols))

        print(df_cols)

        col_indexes = [df_cols.index(f) for f in field_names]
        cols = df.columns[col_indexes]
    else:
        cols = field_names

    if drop_mode:
        return df.drop(cols, axis=1)
    else:
        return df[cols]


def field_list(field_names, drop_mode=False, ignore_case=True):
    f = partial(field_list_func, field_names=field_names, drop_mode=drop_mode, ignore_case=ignore_case)
    return FunctionTransformer(func=f, validate=False)


def days_to_delta_func(df, column_names, base_column):
    res = df.copy()
    base_col_date = pd.to_datetime(df[base_column], errors='coerce')
    for col in column_names:
        days_open = (base_col_date - pd.to_datetime(res[col], errors='coerce')).dropna().dt.days
        res[col] = days_open  # insert is performed by index hence missing records are not written
    return res


def days_to_delta(column_names, base_column):
    f = partial(days_to_delta_func, column_names=column_names, base_column=base_column)
    d2d = FunctionTransformer(func=f, validate=False)
    return d2d
