from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import ModuleException
import logs.logger as log
import pandas as pd

#Frequency/Count Encoder
class FrequencyEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, targetcol = 'ACTION', colnames = [], min_group_size = 1, concat_result_X = True):

        self.colnames = colnames
        self.targetcol = targetcol
        self.min_group_size = min_group_size
        self.merge_result = concat_result_X
        self.learned_values = {}

    def fit(self, X, y = None):

        log.write_log(f'FreqEncode-fit: Started...', log.logging.DEBUG)

        if len(self.colnames) == 0:
            self.colnames = [x for x in X.columns if (x not in self.targetcol) & ('_Kfold' not in x) & ('_FreqEnc' not in x) & ('_svd' not in x) & (x not in ['ROLE_TITLE', 'MGR_ID'])]

        log.write_log(f'FreqEncode-fit: Number of features to encode: {len(self.colnames)}...', log.logging.DEBUG)

        for colname in self.colnames:
            
            msg = f'Target encoded columns \"{colname}\" not avaliable in the dataframe.'
            #assert(colname in X.columns, msg)            
            if not colname in X.columns:
                raise ModuleException('Freq_Enc', msg)
            
            #freq_grp = X.groupby(by = colname)[colname].count()
            freq_grp = X.groupby(by = colname).size()
            
            if self.min_group_size != 1:
                freq_grp[freq_grp < self.min_group_size] = freq_grp[freq_grp < self.min_group_size].sum()   

            self.learned_values[colname] = freq_grp


    def transform(self, X, y = None):

        if len(self.learned_values) == 0:
            raise ModuleException('Freq_Enc', 'Frequency Encoding instance is not fitted yet.')
        
        log.write_log(f'FreqEncode-transform: Started...', log.logging.DEBUG)

        #FreqEnc_col = []
        transformed_X = pd.DataFrame()

        for colname in self.colnames:
            
            if not colname in X.columns:
                raise ModuleException('Freq_Enc', f'Target encoded columns \"{colname}\" not avaliable in the dataframe.')

            if X[colname].dtype != 'object': #type('object')
                raise ModuleException('Freq_Enc', f'\"{colname}\" is not categorical type.')

            if not colname in self.learned_values:
                raise ModuleException('Freq_Enc', f'Frequency Encoding of feature \"{colname}\" is not fitted.')

            lr_value = self.learned_values[colname]
            
            freq_enc_col_name = colname + '_FreqEnc'
            transformed_X[freq_enc_col_name] = X[colname]
            transformed_X[freq_enc_col_name] = X[colname].map(lr_value)
            
            #FreqEnc_col.append(freq_enc_col_name)

        #return X[FreqEnc_col] #default changes get made in the the origina input param

        log.write_log(f'FreqEncode-transform: Number of feature after encoded: {len(transformed_X.columns)}...', log.logging.DEBUG)

        if self.merge_result == True:

            X = pd.concat([X, transformed_X], axis = 1)
            log.write_log(f'FreqEncode-transform: Total number of feature after encode: {len(X.columns)}...', log.logging.DEBUG)

            #X.reset_index(drop = True, inplace = True)
            return X
        else:

            #transformed_X.reset_index(drop = True, inplace = True)
            return transformed_X

    def fit_transform(self, X, y = None):

        if len(self.learned_values) == 0:
            self.fit(X)

        return self.transform(X) 


#=========================== Sample Codes ===========================
"""
#Trial Code
from src.models.feature_eng.FreqEncoding import FrequencyEncoding
import pandas as pd
data = [10,20,30,10,40,30,20,10,50,60,10]
X = pd.DataFrame({'data': data})
X = X.astype(str)
freq_obj = FrequencyEncoding(colnames = ['data'], min_group_size = 1)
#freq_obj.fit(X); freq_obj.transform(X)
freq_obj.fit_transform(X)
freq_obj.learned_values

#Code to understand CountEncoder works
import pandas as pd
import category_encoders as ce
data = [10,20,30,20, 30, 10,40,30,40, 20,10,50,60,10]
X = pd.DataFrame({'data': data})
X = X.astype(str)
cnt = ce.CountEncoder()
X_te = cnt.fit_transform(X)
X_te = pd.concat([X, X_te], axis = 1)
X_te

#With min group_size
# Understand para
# 1. min_group_name: 
#    Internal it create a new column and name is the combination 
#    of all the group value that does not meet the min group condition. 
#    In some case this nae become too long. If so name mention in this group is used 
#2. min_group_size:
#   Min size/len/data points with in each group.
cnt_1 = ce.CountEncoder(min_group_size = 3, min_group_name = 'Min_grp' )
X_t = cnt_1.fit_transform(X)
X_t = pd.concat([X, X_t], axis = 1)
X_t #Output: all the group with size less then min value are grouped and there count is replaced

#Case is there is only one group that does not meet the min value
data = [10,20,30,10,40,30,20,10]
X_1 = pd.DataFrame({'data': data})
X_1= X_1.astype(str)
cnt_1 = ce.CountEncoder(min_group_size = 2)
X_t1 = cnt_1.fit_transform(X_1)
X_t1 = pd.concat([X_1, X_t1], axis = 1)
X_t1 #Output: will be same as no min_group_size paramter

"""