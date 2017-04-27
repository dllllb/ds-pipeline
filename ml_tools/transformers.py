from sklearn.base import BaseEstimator, TransformerMixin


def high_cardinality_zeroing_func(df, threshold, placeholder, columns=None):
    if columns is None:
        cols = df.select_dtypes(include=['object'])
    else:
        cols = columns

    dfc = df.copy()
    for col in cols:
        vc = dfc[col].value_counts()
        dfc.ix[~dfc[col].isin(vc[vc > threshold].index), col] = placeholder
    return dfc


def high_cardinality_zeroing(threshold=49, placeholder='zeroed', columns=None):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a']})
    >>> high_cardinality_zeroing(2).fit_transform(df).A.tolist()
    ['a', 'zeroed', 'zeroed', 'a', 'a']
    """
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(high_cardinality_zeroing_func, threshold=threshold, placeholder=placeholder, columns=None)
    return FunctionTransformer(func=f, validate=False)


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
            entries = df[col].value_counts(dropna=False).index
            self.vc[col] = dict(zip(entries, range(len(entries))))

        return self

    def transform(self, X):
        import numpy as np
        res = X.copy()
        for col, mapping in self.vc.items():
            res[col] = res[col].map(lambda x: mapping.get(x, mapping.get(np.nan, 0)))
        return res


def build_categorical_feature_encoder(df, y, col, target_label):
    vc = df[col].value_counts(dropna=False)
    true_vc = df[y == target_label][col].value_counts(dropna=False)
    entries = (true_vc / vc).sort_values(ascending=False).index
    encoder = dict(zip(entries, range(len(entries))))
    return col, encoder


class TargetShareCountEncoder(BaseEstimator, TransformerMixin):
    """
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    >>> TargetShareCountEncoder().fit_transform(df, np.array([0, 1, 1, 1, 0, 1, 0])).A.tolist()
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

        self.vc = dict(Parallel(n_jobs=self.n_jobs)(
            delayed(build_categorical_feature_encoder)(df, y, col, self.target_label)
            for col in columns
        ))

        return self

    def transform(self, X):
        import numpy as np
        res = X.copy()
        for col, mapping in self.vc.items():
            res[col] = res[col].map(lambda x: mapping.get(x, mapping.get(np.nan, 0)))
        return res


def field_list_func(df, field_names):
    field_names_low_case = map(unicode.lower, field_names)
    df.columns = map(str.lower, df.columns)

    return df[field_names_low_case]


def field_list(field_names):
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(field_list_func, field_names=field_names)
    return FunctionTransformer(func=f, validate=False)


def days_to_delta_func(df, column_names, base_column):
    import numpy as np
    import pandas as pd
    delta = np.timedelta64(1, 'D')
    base_col_date = pd.to_datetime(df[base_column], errors='coerce')
    for col in column_names:
        days_open = ((base_col_date - pd.to_datetime(df[col], errors='coerce')) / delta).astype(np.int16)
        df[col] = days_open
    return df


def days_to_delta(column_names, base_column):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['2015-01-02', '2016-03-20'], 'B': ['2016-02-02', '2016-10-22']})
    >>> days_to_delta(['A'], 'B').fit_transform(df).A.tolist()
    [396, 216]
    """
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(days_to_delta_func, column_names=column_names, base_column=base_column)
    d2d = FunctionTransformer(func=f, validate=False)
    return d2d