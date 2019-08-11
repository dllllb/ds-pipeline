from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from functools import partial


def field_list_func(df, field_names, drop_mode=False, ignore_case=True):
    if ignore_case:
        field_names = list(map(lambda e: e.lower(), field_names))

        df_cols = list(map(lambda e: e.lower(), df.columns))

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
