import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

from dstools.ml.feature_select import FeatureGroupSelector

def test_feature_select():
    boston = load_boston()

    importance_est = RandomForestRegressor(n_estimators=5)
    selector = FeatureGroupSelector(importance_est, pd.DataFrame.corr, .3)

    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    selector.fit(data, boston.target)

    data_tr = selector.transform(data)
    
    print(data.columns)
    print(data_tr.columns)

    print(selector.top_features)