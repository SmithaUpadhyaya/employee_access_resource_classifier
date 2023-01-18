import logging
from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import ModuleException
import logs.logger as log
import pandas as pd
import sys

#Frequency/Count Encoder
class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, targetcol):

        super().__init__()
        self.targetcols = targetcol
        self.learned_values = {}

    def fit(self, X, y = None):

        try:

            for colname in self.targetcols:
            
                msg = 'Target encoded columns \"'+ colname + '\" not avaliable in the dataframe.'
                #assert(colname in X.columns, msg)            
                if not colname in X.columns:
                    raise ModuleException('Freq_Enc', msg, sys)
                
                #freq_grp = X.groupby(by = colname)[colname].count()
                freq_grp = X.groupby(by = colname).size()
                
                self.learned_values[colname] = freq_grp

        #except AssertionError as msg:
        #    raise Exception(f'Module: Freq_Enc- {msg}')
        
        except ModuleException:
            raise
        #except Exception as ex:
        #    log.write_log(f'Freq_Enc: {str(ex)}', log.logging.ERROR)
        #    raise

    def transform(self, X, y = None):

        try:

            if len(self.learned_values) == 0:
                raise ModuleException('Freq_Enc', 'Frequency Encoding instance is not ftted yet.', sys)

            FreqEnc_col = []
            for colname in self.targetcols:
                
                if not colname in X.columns:
                    raise ModuleException('Freq_Enc', f'Target encoded columns \"{colname}\" not avaliable in the dataframe.', sys)

                if X[colname].dtype != 'object':
                    raise ModuleException('Freq_Enc', f'\"{colname}\" is not categorical type.', sys)

                if colname in self.learned_values:
                    raise ModuleException('Freq_Enc', f'Frequency Encoding of feature \"{colname}\" is not fitted.')

                lr_value = self.learned_values[colname]
                freq_enc_col_name = colname + '_FreqEnc'
                X[freq_enc_col_name] = X[colname].map(lr_value)
                FreqEnc_col.append(freq_enc_col_name)

            return X[FreqEnc_col]

        except ModuleException:
            raise
        #except Exception as ex:
        #    log.write_log(f'Freq_Enc: {str(ex)}', log.logging.ERROR)
        #    raise
        

    def fit_transform(self, X, y = None):

        if len(self.learned_values) == 0:
            self.fit(X,y)

        X_return = self.transform(X, y)
        return X_return


#=========================== Sample Codes
"""
#Trial Code
from src.models.feature_eng.FreqEncoding import FrequencyEncoder
import pandas as pd
data = [10,20,30,10,40,30,20,10,50,60,10]
X = pd.DataFrame({'data': data})
X = X.astype(str)
freq_obj = FrequencyEncoder(['data'])
freq_obj.fit(X)
freq_obj.transform(X)

#Code to understand CountEncoder works
import pandas as pd
import category_encoders as ce
data = [10,20,30,10,40,30,20,10,50,60,10]
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
cnt_1 = ce.CountEncoder(min_group_size = 2, min_group_name = 'Min_grp' )
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