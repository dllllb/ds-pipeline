import pandas as pd
import numpy as np

from dstools.ml.transformers import target_mean_encoder
from dstools.ml.transformers import count_encoder
from dstools.ml.transformers import field_list
from dstools.ml.transformers import days_to_delta
from dstools.ml.transformers import high_cardinality_zeroing
from dstools.ml.transformers import multi_class_target_share_encoder
from dstools.ml.transformers import empirical_bayes_encoder
from dstools.ml.transformers import empirical_bayes_vibrant_encoder
from dstools.ml.transformers import empirical_bayes_encoder_normal


def test_multi_class_target_share_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', 'b', np.nan, np.nan, 'b']})
    y = np.array([1, 2, 0, 0, 1, 2, 0, 1, 0])

    pdf = pd.DataFrame({'col': df.A.fillna('nan'), 'target': y})
    pdf["value"] = 1
    pt = pdf.pivot_table(index='col', columns='target', aggfunc='count', fill_value=0)
    counts = pdf.col.value_counts()
    print(pt.divide(counts, axis=0))

    dft = multi_class_target_share_encoder().fit_transform(df, y)
    dft['A'] = df.A
    print(dft)


def test_target_mean_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 0, 1])

    means = y.groupby(df.A.fillna('nan')).mean()
    print(means.sort_values(ascending=False))

    dft = target_mean_encoder().fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_target_mean_encoder_clone():
    from sklearn.base import clone
    t = target_mean_encoder(size_threshold=100)
    t2 = clone(t)
    assert(t2.builder.keywords['size_threshold'] == 100)


def test_parallel_target_mean_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 0, 1])

    dft = target_mean_encoder(n_jobs=2).fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_target_bayes_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 0, 1])

    print('global share: {}'.format(y.mean()))

    means = y.groupby(df.A.fillna('nan')).mean()
    print(means.sort_values(ascending=False))

    dft = empirical_bayes_encoder().fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_bayes_vibrant_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 0, 1])

    print('global share: {}'.format(y.mean()))

    means = y.groupby(df.A.fillna('nan')).mean()
    print(means.sort_values(ascending=False))

    dft = empirical_bayes_vibrant_encoder().fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_target_bayes_encoder_normal_distr():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series(np.arange(7))

    print('global share: {}'.format(y.mean()))

    means = y.groupby(df.A.fillna('nan')).mean()
    print(means.sort_values(ascending=False))

    dft = empirical_bayes_encoder_normal().fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_high_cardinality_zeroing():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a']})
    r = high_cardinality_zeroing(threshold=2).fit_transform(df).A.tolist()
    assert(r == ['a', 'zeroed', 'zeroed', 'a', 'a'])
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'b', 'a', 'a', 'c', 'c']})
    r = high_cardinality_zeroing(top=2).fit_transform(df).A.tolist()
    assert(r == ['a', 'b', 'b', 'b', 'a', 'a', 'zeroed', 'zeroed'])


def test_count_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan]})
    r = count_encoder().fit_transform(df).A.tolist()
    assert (r == [0, 1, 1, 0, 0, 2])


def test_field_list():
    df = pd.DataFrame(np.arange(9).reshape((3, -1)), columns=['A', 'B', 'C'])
    r = field_list(['a', 'b']).transform(df).columns.tolist()
    assert(r == ['A', 'B'])


def test_days_to_delta():
    df = pd.DataFrame({'A': ['2015-01-02', '2016-03-20', '42'], 'B': ['2016-02-02', '2016-10-22', '2016-10-22']})
    r = days_to_delta(['A'], 'B').fit_transform(df).A.fillna(-999).tolist()
    assert(r == [396.0, 216.0, -999.0])
