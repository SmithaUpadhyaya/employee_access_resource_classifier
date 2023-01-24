from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
import logs.logger as log

class CombineFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, targetcol = 'ACTION'):

        self.targetcol = targetcol
        #self.use_columns = use_columns
        
    def combine_cols(self, X):

        transform_X = X.copy()

        self.use_columns = [x for x in transform_X.columns if (x not in ['ROLE_TITLE', 'MGR_ID']) & (x not in self.targetcol)]
        log.write_log(f'CombineFeature: Number of features to combine: {len(self.use_columns)}...', log.logging.DEBUG)

        for c1,c2 in combinations(self.use_columns, 2): #permutations #Number of unique count where same i.e col1_col2 == col2_col1
        
            if (c1 == self.targetcol) | (c2 == self.targetcol):
                continue

            name = "{}_{}".format(c1, c2)
            transform_X[name] = transform_X[c1] + " " + transform_X[c2]

        #transform_X.reset_index(drop = True, inplace = True)
        log.write_log(f'CombineFeature: Total number of after combine: {len(transform_X.columns)}...', log.logging.DEBUG)
        return transform_X

    def fit(self, X, y = None):
        return self
        
    #def fit_transform(self, X, y = None):
    #    return self.combine_cols(X)
    
    def transform(self, X, y = None):
        return self.combine_cols(X)
    