import unittest

import pandas as pd
import numpy as np

from dstools.ml.transformers import MultiClassTargetMeanEncoder, TargetMeanEncoder, MultiClassTargetShareCountEncoder, \
    TargetShareCountEncoder


class TestMultiClassTargetMeanEncoder(unittest.TestCase):
    def test_base(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', 'b', np.nan, np.nan, 'b']})
        y = np.array([1, 2, 0, 0, 1, 2, 0, 1, 0])

        pdf = pd.DataFrame({'col': df.A.fillna('nan'), 'target': y})
        pdf["value"] = 1
        pt = pdf.pivot_table(index='col', columns='target', aggfunc='count', fill_value=0)
        counts = pdf.col.value_counts()
        print(pt.divide(counts, axis=0))

        dft = MultiClassTargetMeanEncoder(n_jobs=1).fit_transform(df, y)
        dft['A'] = df.A
        print(dft)


class TestTargetMeanEncoder(unittest.TestCase):
    def test_base(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
        y = pd.Series([1, 0, 0, 0, 1, 0, 1])

        means = y.groupby(df.A.fillna('nan')).mean()
        print(means.sort_values(ascending=False))

        dft = TargetMeanEncoder(n_jobs=1).fit_transform(df, y)
        dft['A0'] = df.A
        print(dft)


class TestMultiClassTargetShareCountEncoder(unittest.TestCase):
    def test_base(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', 'b', np.nan, np.nan, 'b']})
        y = np.array([1, 2, 0, 0, 1, 2, 0, 1, 0])

        pdf = pd.DataFrame({'col': df.A.fillna('nan'), 'target': y})
        pdf["value"] = 1
        pt = pdf.pivot_table(index='col', columns='target', aggfunc='count', fill_value=0)
        counts = pdf.col.value_counts()
        print(pt.divide(counts, axis=0))

        dft = MultiClassTargetShareCountEncoder(n_jobs=1).fit_transform(df, y)
        dft['A'] = df.A
        print(dft)


class TestTargetShareCountEncoder(unittest.TestCase):
    def test_base(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
        y = pd.Series([1, 0, 0, 0, 1, 0, 1])

        target_share = (y == 1).groupby(df.A.fillna('nan')).mean()
        print(target_share.sort_values(ascending=False))

        dft = TargetShareCountEncoder(n_jobs=1).fit_transform(df, y)
        dft['A0'] = df.A
        print(dft)

    def test_parallel(self):
        df = pd.DataFrame({'A': ['a', 'b', 'b', 'a', 'a', np.nan, np.nan]})
        y = np.array([1, 0, 0, 0, 1, 0, 1])

        dft = TargetShareCountEncoder(n_jobs=1).fit_transform(df, y)
        dft['A0'] = df.A
        print(dft)
