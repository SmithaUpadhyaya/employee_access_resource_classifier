from sklearn.base import BaseEstimator, TransformerMixin
from utils.exception import ModuleException
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# KFold Frequency/Count Encoder
# Will be assigning different Encoder of same group value 
class KFoldFrequencyEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, targetcol, min_group_size = 1, n_fold = 5, random_seed = 2023):

        self.targetcols = targetcol
        self.min_group_size = min_group_size
        self.learned_values = {}
        self.kf = KFold(n_splits = n_fold, shuffle = True, random_state = random_seed)
   
    def fit(self, X):
        return self

    def fit_transform(self, X):

        #Fit train data and transform them.        
        #if len(self.learned_values) == 0: 
        self.learned_values = {}
        
        transformed_X = pd.DataFrame()
        KFold_FE_col = []

        for colname in self.targetcols:
            
            if not colname in X.columns:
                raise ModuleException('KFoldFreq_Enc', f'Encoder columns \"{colname}\" not avaliable in the DataFrame.')

            if X[colname].dtype != 'object':
                raise ModuleException('KFoldFreq_Enc', f'\"{colname}\" is not categorical type.')

            freq_enc_col_name = colname + '_KFoldFreqEnc'
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
            return transformed_X[KFold_FE_col]        
    

    def transform(self, X):

        #This is used when want to transfom test data
        if len(self.learned_values) == 0:
            raise ModuleException('KFoldFreq_Enc', 'KFold Frequency Encoding instance is not fitted yet. Try calling fit_transform first.')
            
        transformed_X = pd.DataFrame()

        for colname in self.targetcols:

            if not colname in X.columns:
                raise ModuleException('KFoldFreq_Enc', f'Encoded columns \"{colname}\" not avaliable in the dataframe.')

            if not colname in self.learned_values:
                raise ModuleException('KFoldFreq_Enc', f'Frequency Encoding of feature \"{colname}\" is not fitted.')

            lr_value = self.learned_values[colname]
                
            freq_enc_col_name = colname + '_KFoldFreqEnc'
            transformed_X[freq_enc_col_name] = X[colname]
            transformed_X[freq_enc_col_name] = X[colname].map(lr_value)
        
        return transformed_X

#=========================== Sample Codes
"""
#Trial Code
from src.models.feature_eng.KFoldFreqEncoding import KFoldFrequencyEncoding
import pandas as pd
data = [10,20,30,10,40,30,20,10,50,60,10]
X = pd.DataFrame({'data': data})
X = X.astype(str)
Kfreq_obj = KFoldFrequencyEncoding(targetcol = ['data'], min_group_size = 1)
Kfreq_obj.fit_transform(X)
Kfreq_obj.learned_values

X_test = pd.DataFrame({'data': [10,20,30,40,50,60]})
X_test = X_test.astype(str)
Kfreq_obj.transform(X_test)
"""