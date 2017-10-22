import pandas as pd
import numpy as np

from dstools.ml.transformers import TargetMeanEncoder
from dstools.ml.transformers import CountEncoder
from dstools.ml.transformers import field_list
from dstools.ml.transformers import days_to_delta
from dstools.ml.transformers import HighCardinalityZeroing
from dstools.ml.transformers import MultiClassTargetShareEncoder
from dstools.ml.transformers import TargetEmpyricalBayesEncoder


def test_multi_class_target_share_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', 'b', np.nan, np.nan, 'b']})
    y = np.array([1, 2, 0, 0, 1, 2, 0, 1, 0])

    pdf = pd.DataFrame({'col': df.A.fillna('nan'), 'target': y})
    pdf["value"] = 1
    pt = pdf.pivot_table(index='col', columns='target', aggfunc='count', fill_value=0)
    counts = pdf.col.value_counts()
    print(pt.divide(counts, axis=0))

    dft = MultiClassTargetShareEncoder().fit_transform(df, y)
    dft['A'] = df.A
    print(dft)


def test_target_mean_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 0, 1])

    means = y.groupby(df.A.fillna('nan')).mean()
    print(means.sort_values(ascending=False))

    dft = TargetMeanEncoder().fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_parallel_target_mean_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 0, 1])

    dft = TargetMeanEncoder(n_jobs=2).fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_target_bayes_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
    y = pd.Series([1, 0, 0, 0, 1, 0, 1])

    print('global share: {}'.format(y.mean()))

    means = y.groupby(df.A.fillna('nan')).mean()
    print(means.sort_values(ascending=False))

    dft = TargetEmpyricalBayesEncoder().fit_transform(df, y)
    dft['A0'] = df.A
    print(dft)


def test_high_cardinality_zeroing():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a']})
    r = HighCardinalityZeroing(2).fit_transform(df).A.tolist()
    assert(r == ['a', 'zeroed', 'zeroed', 'a', 'a'])
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', 'c', 'c']})
    r = HighCardinalityZeroing(top=2).fit_transform(df).A.tolist()
    assert(r == ['a', 'b', 'b', 'a', 'a', 'zeroed', 'zeroed'])


def test_count_encoder():
    df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan]})
    r = CountEncoder().fit_transform(df).A.tolist()
    assert (r == [0, 1, 1, 0, 0, 2])


def test_field_list():
    df = pd.DataFrame(np.arange(9).reshape((3, -1)), columns=['A', 'B', 'C'])
    r = field_list(['a', 'b']).transform(df).columns.tolist()
    assert(r == ['A', 'B'])


def test_days_to_delta():
    df = pd.DataFrame({'A': ['2015-01-02', '2016-03-20', '42'], 'B': ['2016-02-02', '2016-10-22', '2016-10-22']})
    r = days_to_delta(['A'], 'B').fit_transform(df).A.fillna(-999).tolist()
    assert(r == [396.0, 216.0, -999.0])