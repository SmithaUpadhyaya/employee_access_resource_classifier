from sklearn.base import BaseEstimator, TransformerMixin
from utils.read_utils import read_yaml_key
import logs.logger as log
import pandas as pd
import numpy as np

class RandomCatagoryEncode(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.params = read_yaml_key('featurize.random_catagory_encode')
        self.targetcol = self.params['targetcol']

    def assign_rnd_integer(self, X):

        np.random.seed(self.params['random_seed'])

        colnames = [x for x in X.columns if (x not in self.targetcol) & ('_Kfold' not in x) & ('_FreqEnc' not in x) & ('_svd' not in x) & ('_rnd_int_enc' not in x) & (x not in ['ROLE_TITLE', 'MGR_ID'])]

        log.write_log(f'RandomCatagoryEncode: Number of features to encode: {len(colnames)}...', log.logging.DEBUG)

        for col in colnames:
            
            if not col in self.params['columns']:
                continue
            
            col_name = col + "_rnd_int_enc"
            unique_vals = X[col].unique()
            labels = np.array(list(range(len(unique_vals))))
            np.random.shuffle(labels)
            mapping = pd.DataFrame({col: unique_vals, col_name: labels})
            
            X = X.merge(mapping, on = col, how = 'inner')

        log.write_log(f'RandomCatagoryEncode: Total number of after encode: {len(X.columns)}...', log.logging.DEBUG)

        return X

    def fit(self, X, y = None):
        return self
        
    #def fit_transform(self, X, y = None):
    #    return self.combine_cols(X)
    
    def transform(self, X, y = None):
        return self.assign_rnd_integer(X)
    