import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial import distance
from scipy.cluster import hierarchy

class FeatureGroupSelector(BaseEstimator, TransformerMixin):
    def __init__(self, importance_estimator, feature_distance_calculator, cluster_threshold):
        self.taken_columns = None
        self.importance_estimator = importance_estimator
        self.feature_distance_calculator = feature_distance_calculator
        self.cluster_threshold = cluster_threshold

    def fit(self, df, y):
        dm = self.feature_distance_calculator(df)
        li = hierarchy.linkage(distance.squareform(1-np.abs(dm)))

        clust = hierarchy.fcluster(li, t=.3, criterion='distance')
        f_clusters = pd.Series(clust, index=dm.columns).sort_values()

        self.importance_estimator.fit(df, y)
        f_weights = pd.Series(self.importance_estimator.feature_importances_, index=df.columns)

        not_clustered = f_weights.index[~f_weights.index.isin(f_clusters.index)]
        nc_cluster_names = ['nc' + str(n) for n in range(len(not_clustered))]
        not_clustered_clusters = pd.Series(nc_cluster_names, index=not_clustered, name='cluster')
        f_clusters_full = pd.concat([f_clusters.astype(str), not_clustered_clusters])

        fic = pd.DataFrame({'cluster': f_clusters_full, 'weight': f_weights})
        cluster_means = fic.groupby('cluster')['weight'].mean().rename('mean_weight')
        ficm = fic.join(
            cluster_means, on='cluster'
        ).sort_values(['mean_weight', 'cluster'], ascending=False)
        ficm = ficm.reset_index().set_index(['cluster', 'index'])

        wfct = ficm.reset_index().set_index('index')
        self.top_features = wfct.loc[
            wfct.groupby('cluster').weight.idxmax()
        ].sort_values('weight', ascending=False)

        return self

    def transform(self, df):
        return df.copy()[self.top_features.index]
