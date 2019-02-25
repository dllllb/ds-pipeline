import pandas as pd
import numpy as np

from dstools.ml.transformers import field_list
from dstools.ml.transformers import days_to_delta


def test_field_list():
    df = pd.DataFrame(np.arange(9).reshape((3, -1)), columns=['A', 'B', 'C'])
    r = field_list(['a', 'b']).transform(df).columns.tolist()
    assert(r == ['A', 'B'])


def test_days_to_delta():
    df = pd.DataFrame({'A': ['2015-01-02', '2016-03-20', '42'], 'B': ['2016-02-02', '2016-10-22', '2016-10-22']})
    r = days_to_delta(['A'], 'B').fit_transform(df, ).A.fillna(-999).tolist()
    assert(r == [396.0, 216.0, -999.0])
