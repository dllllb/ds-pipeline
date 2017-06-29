from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


def zeroing_candidates(data, threshold, top):
    vc = data.value_counts()
    candidates = set(vc[vc <= threshold].index).union(set(vc[top:].index))
    return candidates


class HighCardinalityZeroing(BaseEstimator, TransformerMixin):
    """
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a']})
    >>> HighCardinalityZeroing(2).fit_transform(df).A.tolist()
    ['a', 'zeroed', 'zeroed', 'a', 'a']
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', 'c', 'c']})
    >>> HighCardinalityZeroing(top=2).fit_transform(df).A.tolist()
    ['a', 'b', 'b', 'a', 'a', 'zeroed', 'zeroed']
    """

    def __init__(self, threshold=1, top=10000, placeholder='zeroed', columns=None, n_jobs=1):
        self.zero_categories = dict()
        self.threshold = threshold
        self.top = top
        self.placeholder = placeholder
        self.columns = columns
        self.n_jobs = n_jobs

    def fit(self, df, y=None):
        from sklearn.externals.joblib import Parallel, delayed

        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        self.zero_categories = dict(zip(columns, Parallel(n_jobs=self.n_jobs)(
            delayed(zeroing_candidates)(df[col], self.threshold, self.top)
            for col in columns
        )))

        return self

    def transform(self, X):
        res = X.copy()
        for col, candidates in self.zero_categories.items():
            res[col] = res[col].map(lambda x: self.placeholder if x in candidates else x)
        return res


def df2dict():
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(
        lambda x: x.to_dict(orient='records'), validate=False)


class CountEncoder(BaseEstimator, TransformerMixin):
    """
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan]})
    >>> CountEncoder().fit_transform(df).A.tolist()
    [0, 1, 1, 0, 0, 2]
    """

    def __init__(self):
        self.vc = dict()

    def fit(self, df, y=None):
        for col in df.select_dtypes(include=['object']):
            # don't use value_counts(dropna=True)!!!
            # in case if joblib n_jobs > 1 the behavior of np.nan key is not stable
            entries = df[col].replace(np.nan, 'nan').value_counts()
            entries = entries.sort_values(ascending=False).index
            self.vc[col] = dict(zip(entries, range(len(entries))))

        return self

    def transform(self, X):
        res = X.copy()
        for col, mapping in self.vc.items():
            res[col] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))
        return res


def build_categorical_feature_encoder(category_series, category_labels):
    # don't use value_counts(dropna=True)!!!
    # in case if joblib n_jobs > 1 the behavior of np.nan key is not stable
    shares = pd.Series(category_labels).groupby(category_series.fillna('nan')).mean()
    entries = shares.sort_values(ascending=False).index

    encoder = dict(zip(entries, range(len(entries))))
    return encoder


class TargetShareCountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_label=1, columns=None, n_jobs=1):
        self.vc = dict()
        self.target_label = target_label  # target_label param is deprecated
        self.columns = columns
        self.n_jobs = n_jobs

    def fit(self, df, y):
        from sklearn.externals.joblib import Parallel, delayed

        self.target_label = pd.Series(y).value_counts().index[1]

        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        self.vc = dict(zip(columns, Parallel(n_jobs=self.n_jobs)(
            delayed(build_categorical_feature_encoder)(df[col], y == self.target_label)
            for col in columns
        )))

        return self

    def transform(self, df):
        res = df.copy()
        for col, mapping in self.vc.items():
            res[col] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))
        return res


class MultiClassTargetShareCountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, n_jobs=1):
        self.class_encodings = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.columns = None

    def fit(self, df, y):
        from sklearn.externals.joblib import Parallel, delayed

        encoded_classes = pd.Series(y).value_counts().index[1:]

        if self.columns is None:
            self.columns = df.select_dtypes(include=['object'])

        for cl in encoded_classes:
            vc = dict(zip(self.columns, Parallel(n_jobs=self.n_jobs)(
                delayed(build_categorical_feature_encoder)(df[col], y == cl)
                for col in self.columns
            )))
            self.class_encodings[cl] = vc

        return self

    def transform(self, df):
        res = df.copy()
        for cls, cols in self.class_encodings.items():
            for col, mapping in cols.items():
                res['{}_{}'.format(col, cls)] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))

        res = res.drop(self.columns, axis=1)
        return res


def build_categorical_feature_encoder_mean(column, target, reg):
    global_mean = target.mean()
    col_dna = column.fillna('nan')
    means = target.groupby(col_dna).mean()
    counts = col_dna.groupby(col_dna).count()
    if reg is None:
        reg = counts.mean()*.2
    means_reg = means * counts + reg * global_mean
    entries = means_reg.sort_values(ascending=False).index

    encoder = dict(zip(entries, range(len(entries))))
    return encoder


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, n_jobs=1, reg=None):
        self.vc = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.reg = reg

    def fit(self, df, y):
        from sklearn.externals.joblib import Parallel, delayed

        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        self.vc = dict(zip(columns, Parallel(n_jobs=self.n_jobs)(
            delayed(build_categorical_feature_encoder_mean)(df[col], y, self.reg)
            for col in columns
        )))

        return self

    def transform(self, df):
        res = df.copy()
        for col, mapping in self.vc.items():
            res[col] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))
        return res


class MultiClassTargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, n_jobs=1, reg=None):
        self.class_encodings = dict()
        self.columns = columns
        self.n_jobs = n_jobs
        self.reg = reg

    def fit(self, df, y):
        from sklearn.externals.joblib import Parallel, delayed

        encoded_classes = pd.Series(y).value_counts().index[1:]

        if self.columns is None:
            self.columns = df.select_dtypes(include=['object'])

        for cl in encoded_classes:
            vc = dict(zip(self.columns, Parallel(n_jobs=self.n_jobs)(
                delayed(build_categorical_feature_encoder_mean)(df[col], pd.Series(y == cl), self.reg)
                for col in self.columns
            )))
            self.class_encodings[cl] = vc

        return self

    def transform(self, df):
        res = df.copy()
        for cls, cols in self.class_encodings.items():
            for col, mapping in cols.items():
                res['{}_{}'.format(col, cls)] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))

        res = res.drop(self.columns, axis=1)
        return res


def field_list_func(df, field_names, drop_mode=False, ignore_case=True):
    if ignore_case:
        field_names = map(unicode, field_names)
        field_names = map(unicode.lower, field_names)

        df_cols = map(unicode, df.columns)
        df_cols = map(unicode.lower, df_cols)

        col_indexes = [df_cols.index(f) for f in field_names]
        cols = df.columns[col_indexes]
    else:
        cols = field_names

    if drop_mode:
        return df.drop(cols, axis=1)
    else:
        return df[cols]


def field_list(field_names, drop_mode=False, ignore_case=True):
    """
    >>> df = pd.DataFrame(np.arange(9).reshape((3, -1)), columns=['A', 'B', 'C'])
    >>> field_list(['a', 'b']).transform(df).columns.tolist()
    ['A', 'B']
    """
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(field_list_func, field_names=field_names, drop_mode=drop_mode, ignore_case=ignore_case)
    return FunctionTransformer(func=f, validate=False)


def days_to_delta_func(df, column_names, base_column):
    res = df.copy()
    base_col_date = pd.to_datetime(df[base_column], errors='coerce')
    for col in column_names:
        days_open = (base_col_date - pd.to_datetime(res[col], errors='coerce')).dropna().dt.days
        res[col] = days_open # insert is performed by index hence missing records are not written
    return res


def days_to_delta(column_names, base_column):
    """
    >>> df = pd.DataFrame({'A': ['2015-01-02', '2016-03-20', '42'], 'B': ['2016-02-02', '2016-10-22', '2016-10-22']})
    >>> days_to_delta(['A'], 'B').fit_transform(df).A.fillna(-999).tolist()
    [396.0, 216.0, -999.0]
    """
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(days_to_delta_func, column_names=column_names, base_column=base_column)
    d2d = FunctionTransformer(func=f, validate=False)
    return d2d
