from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.model_selection import StratifiedKFold
from utils.exception import ModuleException
from utils.read_utils import read_yaml_key
from sklearn.model_selection import KFold
import logs.logger as log
import pandas as pd
import numpy as np

#from sklearn.model_selection import KFold
#import copy #Create a deep copy

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self):

        self.params = read_yaml_key('featurize.ktarget_enc')
        self.colnames = self.params['columns']
        self.targetName = self.params['targetcol']
        self.seed = self.params['random_seed']
        self.n_fold = self.params['n_fold']

        self.merge_result = self.params['concat_result_to_input']
        
        self.global_mean_of_target = 0.0
        self.learned_values = {}  
        self.kf = KFold(n_splits = self.n_fold , shuffle = True, random_state = self.seed)

        #Initial use stratkf. 
        # Fail since 89% of items are unique and they are not avaliable in the fold. 
        # This result is replacing all the encoding with global mean of the target. 
        # As StratifiedKFold generate kfold by preserving the ration of label in the train data.    
        #self.stratkf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = random_seed)
        
    def fit(self, X, y = None):        
        return self

    def fit_transform(self, X, y = None):

        #if len(self.learned_values) == 0:  

        self.learned_values = {}
        
        if (not self.targetName in X.columns) & (type(y) == type(None)):
            #assert(type(y) != type(None), 'Target y is NONE') #assert exception to rasied in the condition is not meet.
            raise ModuleException('KFoldT_Enc', 'target y is None.')
        
        log.write_log(f'TEKFold-fit: Started...', log.logging.DEBUG)  

        if not self.targetName in X.columns:
            X[self.targetName] = y

        if type(y) == type(None):
            y = X[self.targetName]

        self.global_mean_of_target = X[self.targetName].mean()
        
        KFold_TE_col = []
        transformed_X = pd.DataFrame()

        if len(self.colnames) == 0:
            self.colnames = [x for x in X.columns if (x not in self.targetName) & ('_Kfold' not in x) & ('_FreqEnc' not in x) & ('_svd' not in x) & ('_rnd_int_enc' not in x) & (x not in read_yaml_key('featurize.combine_feat.ignore_columns'))] #['ROLE_TITLE', 'MGR_ID']

        log.write_log(f'TEKFold-fit: Number of features to target encode: {len(self.colnames)}...', log.logging.DEBUG)

        for colname in self.colnames:

            #assert(colname in X.columns, 'Target encoded columns \"'+ colname + '\" not avaliable in the dataframe.')
            if not colname in X.columns:
                raise ModuleException('KFoldT_Enc', 'Target encoded columns \"'+ colname + '\" not avaliable in the dataframe.')

            col_mean_name = colname + '_Kfold_TE'
            KFold_TE_col.append(col_mean_name)

            transformed_X[colname] = X[colname]
            transformed_X[col_mean_name] = X[colname]
            transformed_X[col_mean_name] = np.nan

            for tr_ind, val_ind in self.kf.split(X): #This will return row position of records. self.stratkf(X,y)

                #Step 1: Fetch all the feature for records using train_index row index(tr_ind)
                X_tr, X_val = X.iloc[tr_ind,:], X.iloc[val_ind,:] 
                            
                #Step 2: Compute TE on the train dataset and replace the value in validation dataset. 
                transformed_X.loc[list(X_val.index), col_mean_name] = X_val[colname].map(X_tr.groupby(colname)[self.targetName].mean()) 

                #Step 3: Calculate mean of target_col in X_tr dataset, and fill missing value in the X_val. 
                X_tr_targetcol_mean = X_tr[self.targetName].mean()
                transformed_X.loc[list(X_val[X_val.isna()].index), col_mean_name] = X_tr_targetcol_mean
            
            
            #Fill the missing value the global mean of the the target columns
            transformed_X[col_mean_name].fillna(self.global_mean_of_target, inplace = True) 

            #For test dataset consider the mean value of TE for each catagory in the groupby 
            self.learned_values[colname] = transformed_X[[colname, col_mean_name]].groupby(colname)[col_mean_name].mean()

        log.write_log(f'TEKFold-fit: Number of feature after target encoded: {len(KFold_TE_col)}...', log.logging.DEBUG)
        
        if self.merge_result == True:
            
            X = pd.concat([X, transformed_X[KFold_TE_col]], axis = 1)
            log.write_log(f'TEKFold-fit: Total number of feature after target encode: {len(X.columns)}...', log.logging.DEBUG)            
            
            #X.reset_index(drop = True, inplace = True)            
            return X

        else:   
                     
            #transformed_X.reset_index(drop = True, inplace = True)            
            return transformed_X[KFold_TE_col]

    def transform(self, X, y = None):
                       
        #This is used when want to transfom test data
        if len(self.learned_values) == 0:
            raise ModuleException('KFoldT_Enc', 'KFold Target Encoding instance is not fitted yet. Try calling fit_transform first.')             

        log.write_log(f'TEKFold-transform: Started...', log.logging.DEBUG) 

        #To use the global target mean in case of new catagory in the column 
        col_mean = self.global_mean_of_target
        KFold_TE_col = []
        transformed_X = pd.DataFrame()
        
        for colname in self.colnames:

            #assert(colname in X.columns, 'Target encoded feature '+ colname + ' not avaliable in the dataframe')
            if not colname in X.columns:
                raise ModuleException('KFoldT_Enc', 'Target encoded columns \"'+ colname + '\" not avaliable in the dataframe.')

            col_mean_name = colname + '_Kfold_TE'
            transformed_X[col_mean_name] = X[colname]

            KFold_TE_col.append(col_mean_name)

            #Replace the mean value of each groupby column value
            #mean =  X[[colname, col_mean_name]].groupby(colname)[col_mean_name].mean()#.reset_index() #Get the mean of each catagory of the column
            mean = self.learned_values[colname]
            transformed_X[col_mean_name] = X[colname].map(mean)
            
            #Fill the missing value the global mean of the the column
            transformed_X[col_mean_name].fillna(col_mean, inplace = True) 

        log.write_log(f'TEKFold-transform: Number of feature after target encoded: {len(transformed_X.columns)}...', log.logging.DEBUG)
        
        if self.merge_result == True:

            X = pd.concat([X, transformed_X], axis = 1)
            log.write_log(f'TEKFold-transform: Total number of feature after target encode: {len(X.columns)}...', log.logging.DEBUG)
            
            #X.reset_index(drop = True, inplace = True)            
            return X        
        else:
            
            #transformed_X.reset_index(drop = True, inplace = True)            
            return transformed_X

#===================================================================================
#Code to test the logic
"""
for tr_ind, val_ind in te_obj.stratkf.split(X,y):
    X_tr, X_val = X.iloc[tr_ind,:], X.iloc[val_ind,:]
    print(X_tr)
    print(X_val)
    X_val[colname].map(X_tr.groupby(colname)[targetName].mean())
    X_tr[targetName].mean()
    break

import pandas as pd
data = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
label = [1,0,0,0,1,0,1,1,0,0,0,0,1,1,1]
X = pd.DataFrame({'data': data, 'label': label})
y = X['label']
from src.models.feature_eng.TE_KFold import KFoldTargetEncoder
te_obj = KFoldTargetEncoder(['label'], 'data', 2)
te_obj.transform(X, y)
X
X.groupby(by = 'label').mean()
data_test = [1,0]
X_test = pd.DataFrame({'label': data_test})
te_obj.transform(X_test)
X_test

from sklearn.model_selection import StratifiedKFold
stratkf = StratifiedKFold(n_splits = 2, shuffle = True, random_state = 43)
targetName = 'data'
colname =  'label'
#targetName = 'label'
#colname = 'data'
col_mean_name = colname + '_' + 'Kfold_Target_Enc'
mean_of_target = X[targetName].mean()
i = 1               
for tr_ind, val_ind in stratkf.split(X, y): #This will return row position of records  

    print(f'Fold: {i}, tr_ind: {tr_ind}, val_ind:{val_ind}')
    i = i + 1
    X_tr, X_val = X.iloc[tr_ind,:], X.iloc[val_ind,:]
    X.loc[list(X_val.index), col_mean_name] = X_val[colname].map(X_tr.groupby(colname)[targetName].mean()) 

learned_values = {}
learned_values[colname] = X[[colname, col_mean_name]].groupby(colname)[col_mean_name].mean()
"""