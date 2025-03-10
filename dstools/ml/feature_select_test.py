import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

from dstools.ml.feature_select import FeatureGroupSelector

def test_feature_select():
    california = fetch_california_housing()

    importance_est = RandomForestRegressor(n_estimators=5)
    selector = FeatureGroupSelector(importance_est, pd.DataFrame.corr, .3)

    data = pd.DataFrame(california.data, columns=california.feature_names)
    selector.fit(data, california.target)

    data_tr = selector.transform(data)
    
    print(data.columns)
    print(data_tr.columns)

    print(selector.top_features)
