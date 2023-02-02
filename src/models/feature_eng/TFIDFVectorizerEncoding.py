from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import permutations, combinations
from sklearn.decomposition import TruncatedSVD
from utils.exception import ModuleException
from utils.read_utils import read_yaml_key
import logs.logger as log
import pandas as pd

class TFIDFVectorizerEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, concat_result_X = True):  
        
        self.params = read_yaml_key('featurize.tfidf')
        self.targetcol = self.params['targetcol']
        self.combine_columns_required = self.params['combine_columns_required']
        
        self.merge_result = concat_result_X

        self.dict_Vectorizer = {}
        self.dict_dim_reduction = {}
        
    def combine_cols(self, dataset, columns):

        if self.combine_columns_required == True:

            log.write_log(f'TFIDF: Combine features started: {len(columns)}...', log.logging.DEBUG)

            for c1, c2 in combinations(columns, 2): #permutations #Number of unique count where same i.e col1_col2 == col2_col1
            
                if (c1 == self.targetcol) | (c2 == self.targetcol):
                    continue

                name = "{}_{}".format(c1, c2)

                #From EDA: 
                # 1: Skip combination of ROLE_FAMILY and ROLE_CODE
                #if (c1 in ['ROLE_FAMILY', 'ROLE_CODE']) & (c2 in ['ROLE_FAMILY', 'ROLE_CODE']):
                #    continue
                
                ### 2: Skip any combination of feature with  RESOURCE
                ##if (c1 == "RESOURCE") | (c2 == "RESOURCE"):
                ##    continue
                
                #3: skip all other combination which are not define in "use_column"  
                #if len(self.params['use_features']) > 0:
                #    if name not in self.params['use_features']:
                #        continue               
                
                dataset[name] = dataset[c1] + " " + dataset[c2]

        return dataset

    def get_col_interactions_svd(self, dataset, istraining):

        log_code = "TFIDF-"+ "fit" if istraining else "transform"

        colnames = self.params['columns']
        if len(colnames) == 0: 
            colnames = [x for x in dataset.columns if (x not in self.targetcol) & ('_Kfold' not in x) & ('_FreqEnc' not in x) & ('_svd' not in x) & ('_rnd_int_enc' not in x)& (x not in ['ROLE_TITLE', 'MGR_ID'])] 
        
        log.write_log(f'{log_code}: Number of features to consider for vectorize: {len(colnames)}...', log.logging.DEBUG)

        permutat_cnt = 0
        for col1, col2 in permutations(colnames, 2): #dataset.columns
        
            if (col1 == self.targetcol) | (col2 == self.targetcol):
                continue

            data = self.extract_col_interaction(dataset, col1, col2, istraining)

            if type(data) == type(None):
                continue

            #new_dataset will return the encoding for unique value for the combination. 
            #Will use merge to merge them to the X train dataset.

            #Merge records with main X dataset
            #Merge the extracted interaction data about col1 to main dataset by joining them with there key.
            #Reason to do this is we want to merge the calculated interaction data to respective col1 in main dataset
            dataset = dataset.merge(data, on = col1, how = 'inner')
            permutat_cnt += 1

        log.write_log(f'{log_code}: Number of permutations for features: {len(colnames)} is {permutat_cnt}', log.logging.DEBUG)
        
        return dataset

    def extract_col_interaction(self, dataset, col1, col2, istraining):

        key = col1 + "_" + col2
        
        #params = kwargs['params']
        data = dataset.groupby([col1])[col2].agg(lambda x: " ".join(x))

        if key in self.dict_Vectorizer:

            vectorizer = self.dict_Vectorizer[key]

        else:

            if istraining == False:
                return None

            vectorizer = TfidfVectorizer(lowercase = False)            
            self.dict_Vectorizer[key] = vectorizer.fit(data) #Save the fitted vectorizer obj, which will be used in the transform stage
        
        data_X = vectorizer.transform(data)

        if key in self.dict_dim_reduction:
            dim_red = self.dict_dim_reduction[key] 
        else:
            dim_red = TruncatedSVD(n_components = self.params['dim_reduction'], random_state = self.params['random_seed'])

            #Save the fitted dimension reduction obj, which will be used in the transform stage
            dim_red = dim_red.fit(data_X) 

            if dim_red.explained_variance_ratio_[0] >= self.params['var_explained']:

                self.dict_dim_reduction[key] = dim_red.fit(data_X) 

            #Save only those combination that explain the varaince more then 90%
            else:
            
                del [self.dict_Vectorizer[key]]
                return None

        data_X = dim_red.transform(data_X)
        
        col_no = self.params['dim_reduction']
        col_names = []

        if col_no == 1:
            col_names.append(col1 + "_svd_{}_{}".format(col2, 'tf'))
        else:  
            for i in range(col_no):
                col_names.append(col1 + "_svd_{}_{}_{}".format(col2, 'tf', i))
        
        data_X = pd.DataFrame( data_X, columns = col_names)
        
        #Output: Will be for each unique values of col1 will have calculated vectorizer of interacted col2
        result = pd.DataFrame()
        result[col1] = data.index.values
        result = pd.concat([result, data_X], axis = 1) 
        #result[col1 + "_{}_{}_svd".format(col2, 'cv' if feature_code == 'count_vector' else 'tf')] = data_X.ravel()

        return result

    def encode(self, X, istraining):

        log_code = "TFIDF-"+ "fit" if istraining else "transform"

        col_use = [x for x in X.columns if not x in ['ROLE_TITLE', 'MGR_ID']]
        X = X[col_use]

        X = self.combine_cols(X, col_use)

        new_dataset = self.get_col_interactions_svd(X, istraining)
        log.write_log(f'{log_code}: Total number of feature after encode: {len(new_dataset.columns)}...', log.logging.DEBUG)

        if self.merge_result == False:

            #Get all the transformed columns
            col_name = [x for x in new_dataset.columns if "svd" in x]
            log.write_log(f'{log_code}: Feature to retrun encode: {len(col_name)}...', log.logging.DEBUG) 
            return new_dataset[col_name] 

        else:

            return new_dataset

#==============================================================================================================

    #def fit(self, X):
    #    return self
        
    def fit_transform(self, X, y = None):

        log.write_log(f'TFIDF-fit: Started...', log.logging.DEBUG) 

        self.dict_Vectorizer = {}
        self.dict_dim_reduction = {}

        return self.encode(X, istraining = True)

    def transform(self, X, y = None):

        if (len(self.dict_Vectorizer) == 0) | (len(self.dict_dim_reduction) == 0):
            raise ModuleException('TFIDF-transform', 'TFIDF Vectorizer instance is not fitted yet. Try calling fit_transform first.')
        
        log.write_log(f'TFIDF-transform: Started...', log.logging.DEBUG) 

        return self.encode(X, istraining = False) 




        
