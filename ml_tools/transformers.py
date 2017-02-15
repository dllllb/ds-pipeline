from sklearn.base import BaseEstimator, TransformerMixin


def high_cardinality_zeroing_func(df, threshold, placeholder):
    dfc = df.copy()
    for col in dfc.select_dtypes(include=['object']):
        vc = dfc[col].value_counts()
        dfc.ix[~dfc[col].isin(vc[vc > threshold].index), col] = placeholder
    return dfc


def high_cardinality_zeroing(threshold=49, placeholder='zeroed'):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a']})
    >>> high_cardinality_zeroing(2).fit_transform(df).A.tolist()
    ['a', 'zeroed', 'zeroed', 'a', 'a']
    """
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(high_cardinality_zeroing_func, threshold=threshold, placeholder=placeholder)
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


def field_list_func(df, field_names):
    field_names_low_case = map(unicode.lower, field_names)
    df.columns = map(str.lower, df.columns)

    return df[field_names_low_case]


def field_list(field_names):
    from sklearn.preprocessing import FunctionTransformer
    from functools import partial
    f = partial(field_list_func, field_names=field_names)
    return FunctionTransformer(func=f, validate=False)
