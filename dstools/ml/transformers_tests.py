import unittest

import pandas as pd
import numpy as np

from dstools.ml.transformers import TargetMeanEncoder, MultiClassTargetShareEncoder


class TestMultiClassTargetShareEncoder(unittest.TestCase):
    def test_base(self):
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


class TestTargetMeanEncoder(unittest.TestCase):
    def test_base(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
        y = pd.Series([1, 0, 0, 0, 1, 0, 1])

        means = y.groupby(df.A.fillna('nan')).mean()
        print(means.sort_values(ascending=False))

        dft = TargetMeanEncoder().fit_transform(df, y)
        dft['A0'] = df.A
        print(dft)

    def test_share(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
        y = pd.Series([1, 0, 0, 0, 1, 0, 1])

        means = y.groupby(df.A.fillna('nan')).mean()
        print(means.sort_values(ascending=False))

        dft = TargetMeanEncoder(true_label=1).fit_transform(df, y)
        dft['A0'] = df.A
        print(dft)

    def test_parallel(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
        y = pd.Series([1, 0, 0, 0, 1, 0, 1])

        dft = TargetMeanEncoder(n_jobs=2).fit_transform(df, y)
        dft['A0'] = df.A
        print(dft)
