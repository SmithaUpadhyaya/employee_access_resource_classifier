from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
import pandas as pd

class CombineFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, targetcol, use_columns):

        self.targetcol = targetcol
        self.use_columns = use_columns
        
    def combine_cols(self, X ):

        for c1,c2 in combinations(self.use_columns, 2): #permutations #Number of unique count where same i.e col1_col2 == col2_col1
        
            if (c1 == self.targetcol) | (c2 == self.targetcol):
                continue

            name = "{}_{}".format(c1, c2)
            X[name] = X[c1] + " " + X[c2]

        return X

    def fit_transform(self, X):

        return self.combine_cols(X)
    
    def transform(self, X):
        return self.combine_cols(X)
    