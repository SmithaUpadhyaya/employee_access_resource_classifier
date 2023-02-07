from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import ModuleException
from utils.read_utils import read_yaml_key
from sklearn.model_selection import KFold
import logs.logger as log
import pandas as pd
import numpy as np

# KFold Frequency/Count Encoder
# Will be assigning different Encoder of same group value 
class KFoldFrequencyEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, concat_result_X = True):

        self.params = read_yaml_key('featurize.fequency_encode')
        self.colnames = self.params['columns']
        self.targetcol = self.params['targetcol']
        self.min_group_size = self.params['min_group_size']  

        self.merge_result = concat_result_X
        self.learned_values = {}

        self.kf = KFold(n_splits = self.params['n_fold'], shuffle = True, random_state = self.params['random_seed'])
   
    def fit(self, X, y = None):
        return self

    def fit_transform(self, X, y = None):

        #Fit train data and transform them.        
        #if len(self.learned_values) == 0: 
        self.learned_values = {}
        
        transformed_X = pd.DataFrame()
        KFold_FE_col = []

        log.write_log(f'KFreqEncode-fit: Started...', log.logging.DEBUG)

        if len(self.colnames ) == 0:
            self.colnames = [x for x in X.columns if (x not in self.targetcol) & ('_Kfold' not in x) & ('_FreqEnc' not in x) & ('_svd' not in x) & ('_rnd_int_enc' not in x) & (x not in read_yaml_key('featurize.combine_feat.ignore_columns'))] #['ROLE_TITLE', 'MGR_ID']

        log.write_log(f'KFreqEncode-fit: Number of features to encode: {len(self.colnames)}...', log.logging.DEBUG)

        for colname in self.colnames:
            
            if not colname in X.columns:
                raise ModuleException('KFoldFreq_Enc', f'Encoder columns \"{colname}\" not avaliable in the DataFrame.')

            if X[colname].dtype != 'object': #type('object')
                raise ModuleException('KFoldFreq_Enc', f'\"{colname}\" is not categorical type.')

            freq_enc_col_name = colname + '_KfoldFreqEnc'
            KFold_FE_col.append(freq_enc_col_name)
            
            transformed_X[colname] = X[colname]
            transformed_X[freq_enc_col_name] = transformed_X[colname]
            transformed_X[freq_enc_col_name] = np.nan

            for tr_ind, val_ind in self.kf.split(X): #This will return row position of records  

                #Step 1: Fetch all the feature for records using train_index row index(tr_ind)
                #print(f'Train: {tr_ind}, Valid: {val_ind}')
                X_tr, X_val = X.iloc[tr_ind,:], X.iloc[val_ind,:]

                #Step 2: Calculate the size for eacg group
                freq_grp = X_tr.groupby(by = colname).size()
        
                if self.min_group_size != 1:
                    freq_grp[freq_grp < self.min_group_size] = freq_grp[freq_grp < self.min_group_size].sum()   

                #Step 3: Transfrom the value in tr_val data's
                transformed_X.loc[list(X_val.index), freq_enc_col_name] = X_val[colname].map(freq_grp)

                #Step 4: Case when the values in the tr_val does not belong to tr_train. In this case the value in tr_val is NaN. So to we shall end the NaN in the tr_val as 1
                transformed_X.loc[list(X_val.index), freq_enc_col_name] = transformed_X.loc[list(X_val.index),freq_enc_col_name].fillna(1)
                    
                
            #Step 5: Calculate the median of column that will be used for transform test dataset. So consider the mode (i,e repeted count value) value each catagory in the groupby 
            transformed_X[KFold_FE_col] = transformed_X[KFold_FE_col].astype(int)  #Change the dtype to int
            #self.temp = transformed_X[[colname, freq_enc_col_name]]
            self.learned_values[colname] = transformed_X[[colname, freq_enc_col_name]].groupby(colname)[freq_enc_col_name].agg(lambda x: pd.Series.mode(x)[0])
        
        log.write_log(f'KFreqEncode-fit: Number of feature after kfold encoded: {len(KFold_FE_col)}...', log.logging.DEBUG)

        if self.merge_result == True:

            X = pd.concat([X, transformed_X[KFold_FE_col]], axis = 1)
            log.write_log(f'KFreqEncode-fit: Total number of feature after kfold encode: {len(X.columns)}...', log.logging.DEBUG)

            #X.reset_index(drop = True, inplace = True)
            return X
        else: 
               
            #transformed_X.reset_index(drop = True, inplace = True)
            return transformed_X[KFold_FE_col]        
    

    def transform(self, X, y = None):

        #This is used when want to transfom test data
        if len(self.learned_values) == 0:
            raise ModuleException('KFoldFreq_Enc', 'KFold Frequency Encoding instance is not fitted yet. Try calling fit_transform first.')
        
        log.write_log(f'KFreqEncode-transform: Started...', log.logging.DEBUG)

        transformed_X = pd.DataFrame()

        for colname in self.colnames:

            if not colname in X.columns:
                raise ModuleException('KFoldFreq_Enc', f'Encoded columns \"{colname}\" not avaliable in the dataframe.')

            if not colname in self.learned_values:
                raise ModuleException('KFoldFreq_Enc', f'Frequency Encoding of feature \"{colname}\" is not fitted.')

            lr_value = self.learned_values[colname]
                
            freq_enc_col_name = colname + '_KfoldFreqEnc'
            transformed_X[freq_enc_col_name] = X[colname]
            transformed_X[freq_enc_col_name] = X[colname].map(lr_value)
        
        log.write_log(f'KFreqEncode-transform: Number of feature after target encoded: {len(transformed_X.columns)}...', log.logging.DEBUG)

        if self.merge_result == True:

            X = pd.concat([X, transformed_X], axis = 1)
            log.write_log(f'KFreqEncode-transform: Total number of feature after target encode: {len(X.columns)}...', log.logging.DEBUG)
            
            #X.reset_index(drop = True, inplace = True)
            return X        
        else:
            
            #transformed_X.reset_index(drop = True, inplace = True)
            return transformed_X

#=========================== Sample Codes
"""
#Trial Code
from src.models.feature_eng.KFoldFreqEncoding import KFoldFrequencyEncoding
import pandas as pd
data = [10,20,30,10,40,30,20,10,50,60,10]
X = pd.DataFrame({'data': data})
X = X.astype(str)
Kfreq_obj = KFoldFrequencyEncoding(colnames = ['data'], min_group_size = 1)
Kfreq_obj.fit_transform(X)
Kfreq_obj.learned_values

X_test = pd.DataFrame({'data': [10,20,30,40,50,60]})
X_test = X_test.astype(str)
Kfreq_obj.transform(X_test)
"""