from sklearn.base import BaseEstimator, TransformerMixin


def zeroing_candidates(data, threshold, top):
    vc = data.value_counts()
    candidates = set(vc[vc <= threshold].index).union(set(vc[top:].index))
    return candidates


class HighCardinalityZeroing(BaseEstimator, TransformerMixin):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a']})
    >>> HighCardinalityZeroing(2).fit_transform(df).A.tolist()
    ['a', 'zeroed', 'zeroed', 'a', 'a']
    >>> import pandas as pd
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
    >>> import pandas as pd
    >>> import numpy as np
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
            entries = df[col].value_counts()
            entries.loc['nan'] = df[col].isnull().sum()
            entries = entries.sort_values(ascending=False).index
            self.vc[col] = dict(zip(entries, range(len(entries))))

        return self

    def transform(self, X):
        res = X.copy()
        for col, mapping in self.vc.items():
            res[col] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))
        return res


def build_categorical_feature_encoder(category_series, category_true_series):
    # don't use value_counts(dropna=True)!!!
    # in case if joblib n_jobs > 1 the behavior of np.nan key is not stable
    vc = category_series.value_counts()
    vc.loc['nan'] = category_series.isnull().sum()
    true_vc = category_true_series.value_counts()
    true_vc['nan'] = category_true_series.isnull().sum()
    entries = (true_vc / vc).sort_values(ascending=False).index

    encoder = dict(zip(entries, range(len(entries))))
    return encoder


class TargetShareCountEncoder(BaseEstimator, TransformerMixin):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    >>> TargetShareCountEncoder(n_jobs=1).fit_transform(df, np.array([0, 1, 1, 1, 0, 1, 0])).A.tolist()
    [2, 0, 0, 2, 2, 1, 1]
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    >>> TargetShareCountEncoder(n_jobs=2).fit_transform(df, np.array([0, 1, 1, 1, 0, 1, 0])).A.tolist()
    [2, 0, 0, 2, 2, 1, 1]
    """

    def __init__(self, target_label=1, columns=None, n_jobs=1):
        self.vc = dict()
        self.target_label = target_label
        self.columns = columns
        self.n_jobs = n_jobs

    def fit(self, df, y):
        from sklearn.externals.joblib import Parallel, delayed

        if self.columns is None:
            columns = df.select_dtypes(include=['object'])
        else:
            columns = self.columns

        self.vc = dict(zip(columns, Parallel(n_jobs=self.n_jobs)(
            delayed(build_categorical_feature_encoder)(df[col], df[y == self.target_label][col])
            for col in columns
        )))

        return self

    def transform(self, X):
        res = X.copy()
        for col, mapping in self.vc.items():
            res[col] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))
        return res


def field_list_func(df, field_names, drop_mode=False):
    if drop_mode:
        field_names = map(unicode, field_names)
        field_names_low_case = map(unicode.lower, field_names)
        df.columns = map(str.lower, df.columns)

        return df.drop(field_names_low_case, axis=1)
    else:
        field_names_low_case = map(unicode.lower, field_names)
        df.columns = map(str.lower, df.columns)

        return df[field_names_low_case]


def field_list(field_names, drop_mode=False):
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(field_list_func, field_names=field_names, drop_mode=drop_mode)
    return FunctionTransformer(func=f, validate=False)


def days_to_delta_func(df, column_names, base_column):
    import pandas as pd
    res = df.copy()
    base_col_date = pd.to_datetime(df[base_column], errors='coerce')
    for col in column_names:
        days_open = (base_col_date - pd.to_datetime(res[col], errors='coerce')).dropna().dt.days
        res[col] = days_open # insert is performed by index hence missing records are not written
    return res


def days_to_delta(column_names, base_column):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['2015-01-02', '2016-03-20', '42'], 'B': ['2016-02-02', '2016-10-22', '2016-10-22']})
    >>> days_to_delta(['A'], 'B').fit_transform(df).A.fillna(-999).tolist()
    [396.0, 216.0, -999.0]
    """
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(days_to_delta_func, column_names=column_names, base_column=base_column)
    d2d = FunctionTransformer(func=f, validate=False)
    return d2d