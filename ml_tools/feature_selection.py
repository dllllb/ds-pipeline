def feature_clusters(features, t=.3, criterion='distance'):
    from scipy.spatial import distance
    from scipy.cluster import hierarchy
    import pandas as pd
    import numpy as np

    # remove non-numeric fields
    fdf = features.select_dtypes(exclude=['object'])

    # remove fields that have only NaN values
    all_null_cols = fdf.isnull().all()
    fdf = fdf.drop(all_null_cols[all_null_cols].index, axis=1)

    # remove fields that have only 0 values
    all_zero_cols = fdf.sum() == 0
    fdf = fdf.drop(all_zero_cols[all_zero_cols].index, axis=1)

    cm = fdf.corr()
    cm[np.abs(cm) > 1] = 1  # fix corr calculation bug

    li = hierarchy.linkage(distance.squareform(1-np.abs(cm)))
    clust = hierarchy.fcluster(li, t=t, criterion=criterion)
    return pd.Series(clust, index=cm.columns)


def weighted_feature_clusters(f_clusters, f_weights):
    import pandas as pd

    not_clustered = f_weights.index[~f_weights.index.isin(f_clusters.index)]
    nc_cluster_names = ['nc' + str(n) for n in range(len(not_clustered))]
    not_clustered_clusters = pd.Series(nc_cluster_names, index=not_clustered, name='cluster')
    f_clusters_full = f_clusters.astype(str).append(not_clustered_clusters)

    fic = pd.DataFrame({'cluster': f_clusters_full, 'weight': f_weights})
    cluster_means = fic.groupby('cluster')['weight'].mean().rename('mean_weight')
    ficm = fic.join(cluster_means, on='cluster').sort_values(['mean_weight', 'cluster'], ascending=False)
    ficm = ficm.reset_index().set_index(['cluster', 'index'])
    return ficm


def top_features_in_cluster(f_clusters, f_weights):
    wfc = weighted_feature_clusters(f_clusters, f_weights)
    wfct = wfc.reset_index().set_index('index')
    top_features = wfct.ix[wfct.groupby('cluster').weight.idxmax()].sort_values('weight', ascending=False)
    return top_features.weight


def xgboost_named_weights(xgb_weights, feature_names):
    import pandas as pd
    empty_scores = [('f'+str(e), 0) for e in range(len(feature_names))]
    scores = dict(empty_scores)
    scores.update(xgb_weights)

    weights = [v for k, v in sorted([(int(k[1:]), v) for k, v in scores.items()], key=lambda e: e[0])]

    feature_weights = pd.Series(weights, index=feature_names).sort_values(ascending=False)
    return feature_weights
