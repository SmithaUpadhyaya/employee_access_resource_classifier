from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import ModuleException
from utils.read_utils import read_yaml_key
import logs.logger as log
import pandas as pd
import numpy as np

class ResourceEncodeByFeature(BaseEstimator, TransformerMixin):

    def __init__(self): #, concat_result_X = True

        self.params = read_yaml_key('featurize.resource_catagory_encode')
        self.merge_result = self.params['concat_result_to_input']
        self.learned_values = {}

    def encode_resource(self, X):

        log.write_log(f'ResourceEncodeByFeature-transform: Started...', log.logging.DEBUG)

        column_to_consider = self.params['column_to_consider']

        transformed_X = pd.DataFrame()

        for colname in column_to_consider: 

            lr_value = self.learned_values[colname]            
            new_col_name = 'resource_'+ colname.lower() + "_enc"
            transformed_X[new_col_name] = X[colname]
            transformed_X[new_col_name] = X[colname].map(lr_value)
            transformed_X[new_col_name].fillna(self.learned_max_values[colname] + 1, inplace = True)

        log.write_log(f'ResourceEncodeByFeature-transform: Number of feature after encoded: {len(transformed_X.columns)}...', log.logging.DEBUG)

        if self.merge_result == True:
            X = pd.concat([X, transformed_X], axis = 1)
            log.write_log(f'ResourceEncodeByFeature-transform: Total number of feature after encode: {len(X.columns)}...', log.logging.DEBUG)
            return X
        else:
            return transformed_X

    def encode_resource_by_feature(self, X):

        self.learned_values = {}
        self.learned_max_values = {} #Used for new value to inferance

        column_to_consider = self.params['column_to_consider']

        log.write_log(f'ResourceEncodeByFeature-fit: Encode RESOURCE using group by by other feature. Started...', log.logging.DEBUG)

        for col in column_to_consider:
            
            log.write_log(f'ResourceEncodeByFeature-fit: Encode RESOURCE features w.r.t to other feature: {col}', log.logging.DEBUG) #EncodeResourceGrpFeature            
            self.learned_values[col] =  X.groupby(col).RESOURCE.nunique()          
            self.learned_max_values[col] = self.learned_values[col].sort_values(ascending = False)[0]

            #new_col_name = 'resource_'+ col.lower() + "_enc"
            #X = X.merge(X.groupby(col).RESOURCE.nunique().reset_index(name = new_col_name), on = col, how = 'inner')

        log.write_log(f'ResourceEncodeByFeature-fit: Completed.', log.logging.DEBUG)

        #return X    

    def fit(self, X, y = None):

        self.encode_resource_by_feature(X)
        
        return self
    
    def transform(self, X, y = None):

        if len(self.learned_values) == 0:
            raise ModuleException('ResourceEncodeByFeature', 'Resource Encode By Feature instance is not fitted yet.')
        
        return self.encode_resource(X)
        