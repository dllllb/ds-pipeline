def binarize(labels):
    from sklearn.preprocessing import LabelBinarizer
    import numpy as np

    y_test_bin = LabelBinarizer().fit_transform(labels)
    if y_test_bin.shape[1] == 1:
        labels_np = np.array(y_test_bin)
        labels_flat = labels_np.reshape(labels_np.size)
        inverse_labels = (labels_flat == 0).astype(np.int8)
        return np.vstack([inverse_labels, labels_flat]).T
    else:
        return y_test_bin


def integral_report(y_test, y_score, target_names):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from pandas import DataFrame
    import numpy as np

    y_test_bin = binarize(y_test)

    support = np.sum(y_test_bin, axis=0)

    output = [
        (
            name,
            roc_auc_score(yt, ys),
            average_precision_score(yt, ys),
            average_precision_score(yt, ys) / (float(sup)/len(y_test)),
            sup
        )
        for yt, ys, name, sup in zip(
            y_test_bin.T,
            y_score.T,
            target_names,
            support
        )
    ]

    output.append(
        (
            'all',
            roc_auc_score(y_test_bin, y_score),
            average_precision_score(y_test_bin, y_score),
            average_precision_score(y_test_bin, y_score) / (np.mean(support)/len(y_test)),
            len(y_test)
        )
    )

    df = DataFrame(output, columns=['class', 'ROC-AUC', 'PR-AUC', 'PR-AUC-Lift', 'support'])
    return df.set_index('class')


def draw_precision_recall_curve(y_score, y_test, target_names, figsize=None):
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import LabelBinarizer
    import matplotlib.pyplot as plt
    import numpy as np

    y_test_bin = LabelBinarizer().fit_transform(y_test)

    plt.figure(figsize=figsize)
    if y_test_bin.shape[1] < 2:
        precision, recall, _ = precision_recall_curve(y_test_bin[:, 0], y_score[:, 1])
        plt.plot(recall, precision, label='Precision-recall curve')
    else:
        for yt, ys, name in zip(y_test_bin.T, y_score.T, target_names):
            precision, recall, _ = precision_recall_curve(yt, ys)
            plt.plot(recall, precision, label='Precision-recall curve of class {0}'.format(name))

    plt.xlim(xmin=.1, xmax=1)
    plt.ylim(ymin=0, ymax=1.1)
    plt.xticks(np.arange(.1, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()


def precision_vs_recall_plot(plot, y_test, y_score, name):
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    thresholds = np.append(thresholds, [0])
    thresholds.sort()
    plot.set_xlabel('Thresholds')
    plot.set_ylabel('Precision', color='red')
    ax2 = plot.twinx()
    ax2.set_ylabel('Recall', color='blue')
    plot.plot(thresholds, precision, label='Precision curve', color='red')
    ax2.plot(thresholds, recall, label='Recall curve', color='blue')
    plot.set_title(name)


def draw_precision_vs_recall(y_score, y_test, target_names, figsize=None, n_cols=2):
    from sklearn.preprocessing import LabelBinarizer
    import itertools
    import matplotlib.pyplot as plt

    y_test_bin = LabelBinarizer().fit_transform(y_test)

    n_plots = y_test_bin.shape[1]
    if n_plots < 2:
        plt.figure(figsize=figsize)
        precision_vs_recall_plot(
            plt.gca(),
            y_test_bin[:, 0],
            y_score[:, 1],
            'Precision vs Recall')
    else:
        _, subplots = plt.subplots(n_plots / n_cols + 1, n_cols, figsize=figsize)
        plt.subplots_adjust(wspace=.4, hspace=.3)
        sp = itertools.chain.from_iterable(subplots)
        for yt, ys, name, sp in zip(y_test_bin.T, y_score.T, target_names, sp):
            precision_vs_recall_plot(sp, yt, ys, name)
    plt.show()


def draw_roc_curve(y_score, y_test, target_names, figsize=None):
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import LabelBinarizer
    import matplotlib.pyplot as plt
    import numpy as np

    y_test_bin = LabelBinarizer().fit_transform(y_test)

    plt.figure(figsize=figsize)
    if y_test_bin.shape[1] < 2:
        fpr, tpr, _ = roc_curve(y_test_bin[:, 0], y_score[:, 1])
        plt.plot(fpr, tpr, label='ROC curve')
    else:
        for yt, ys, name in zip(y_test_bin.T, y_score.T, target_names):
            fpr, tpr, _ = roc_curve(yt, ys)
            plt.plot(fpr, tpr, label='ROC curve of class {0}'.format(name))

    plt.xlim(xmin=.1, xmax=1)
    plt.ylim(ymin=0, ymax=1.1)
    plt.xticks(np.arange(.1, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def feature_statistics_per_class(features, targets, target_names, bins=5):
    from pandas import DataFrame
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.feature_extraction import DictVectorizer
    import numpy as np

    binned_df = (features.div(features.max())*bins).astype(int).astype(str)
    feature_dict = binned_df.to_dict(orient='records')

    dv = DictVectorizer()
    x = dv.fit_transform(feature_dict)
    y = LabelBinarizer().fit_transform(targets)

    feature_count_df = DataFrame(
        np.dot(x.T.todense(), y),
        columns=target_names,
        index=dv.get_feature_names())
    feature_count_norm_df = feature_count_df.div(DataFrame(y, columns=target_names).sum())
    return feature_count_norm_df


def feature_importance_per_class(
    scores,
    feature_names,
    target_names,
    threshold=1e-5,
    max_features_per_class=100
):
    from pandas import DataFrame
    import pandas as pd
    import numpy as np

    feature_filter = np.sum(np.abs(scores), axis=0) >= threshold
    filtered_scores = scores.T[feature_filter].T
    filtered_feature_names = feature_names[feature_filter]

    classes = target_names if (scores.shape[0] > 1) else ['/'.join(target_names)]
    data = [
        (cls, feature, score, abs(score))
        for cls, scores in zip(classes, filtered_scores)
        for feature, score in zip(filtered_feature_names, scores)
    ]

    df = DataFrame(
        data,
        columns=['class', 'feature', 'score', 'absolute_score'],
    )

    ndf = pd.concat([
        rows.sort_values('absolute_score', ascending=False)[:max_features_per_class]
        for _, rows in df.groupby(['class'])
    ]).sort_values(
        ['class', 'absolute_score'],
        ascending=[True, False]
    ).set_index(['class', 'feature'])

    return ndf[ndf['absolute_score'] > threshold].drop('absolute_score', axis=1)


def total_feature_importance_per_class(scores, feature_names, target_names, threshold=1e-5):
    from pandas import DataFrame
    import numpy as np

    feature_filter = np.sum(np.abs(scores), axis=0) >= threshold
    filtered_scores = scores.T[feature_filter].T
    filtered_feature_names = feature_names[feature_filter]

    columns = [c for c in filtered_scores]
    target_cols = target_names.tolist() if (scores.shape[0] > 1) else ['/'.join(target_names)]
    column_names = target_cols + ['importance']

    columns.append(np.sum(np.abs(filtered_scores), axis=0))

    df = DataFrame(
        np.array(columns).T,
        columns=column_names,
        index=filtered_feature_names
    ).sort_values('importance', ascending=False)
    return df