from sklearn.base import BaseEstimator


class CatBoostDecorator(BaseEstimator):
    def __init__(self, est, category_cols=None):
        self.est = est
        self.category_cols = category_cols
        self.cat_col_idx = None

    def num_cat_x(self, X):
        if len(self.cat_col_idx) == 0:
            return X

        ncx = X.copy()

        for idx in self.cat_col_idx:
            ncx.iloc[:, idx] = ncx.iloc[:, idx].map(lambda e: hash(e) % 2 ** 16)

        return ncx

    def fit(self, X, y):
        if self.category_cols is None:
            self.cat_col_idx = [idx for idx, dtype in enumerate(X.dtypes) if dtype == 'object']
        else:
            self.cat_col_idx = [idx for idx, dtype in enumerate(X.dtypes) if dtype in self.category_cols]

        self.est.fit(self.num_cat_x(X), y, cat_features=self.cat_col_idx)
        return self

    def predict_proba(self, X):
        return self.est.predict_proba(self.num_cat_x(X))

    def predict(self, X):
        return self.est.predict(self.num_cat_x(X))
