from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from utils.exception import ModuleException
import pandas as pd
import numpy as np

#from sklearn.model_selection import KFold
#import copy #Create a deep copy

class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, colnames, targetName, n_fold = 5, random_seed = 2023):

        self.colnames = colnames
        self.targetName = targetName
        #self.n_fold = n_fold
        #self.seed = random_seed

        self.global_mean_of_target = 0.0
        self.learned_values = {}        
        self.stratkf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = random_seed)
        #self.kf = KFold(n_splits = n_fold, shuffle = True, random_state = random_seed)

    def fit(self, X, y = None):        
        return self

    def transform(self, X, y = None):

        if len(self.learned_values) == 0:   

            if (not self.targetName in X.columns) & (type(y) == type(None)):
                #assert(type(y) != type(None), 'Target y is NONE') #assert exception to rasied in the condition is not meet.
                raise ModuleException('KFoldT_Enc', 'target y is None.')
            
            if not self.targetName in X.columns:
                X[self.targetName] = y

            if type(y) == type(None):
                y = X[self.targetName]

            self.global_mean_of_target = X[self.targetName].mean()
            
            KFold_TE_col = []
            transformed_X = pd.DataFrame()

            for colname in self.colnames:

                #assert(colname in X.columns, 'Target encoded columns \"'+ colname + '\" not avaliable in the dataframe.')
                if not colname in X.columns:
                    raise ModuleException('KFoldT_Enc', 'Target encoded columns \"'+ colname + '\" not avaliable in the dataframe.')

                col_mean_name = colname + '_Kfold_TE'
                KFold_TE_col.append(col_mean_name)

                transformed_X[colname] = X[colname]
                transformed_X[col_mean_name] = X[colname]
                transformed_X[col_mean_name] = np.nan

                for tr_ind, val_ind in self.stratkf.split(X, y): #This will return row position of records  

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

                return transformed_X[KFold_TE_col]
               
        else:               
            
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