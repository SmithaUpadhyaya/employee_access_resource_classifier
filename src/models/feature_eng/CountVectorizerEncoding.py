from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import permutations, combinations
from sklearn.decomposition import TruncatedSVD
from utils.exception import ModuleException
import pandas as pd

class CountVectorizerEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, targetcol, params):

        self.targetcol = targetcol
        self.params = params

        self.dict_Vectorizer = {}
        self.dict_dim_reduction = {}
        
    def combine_cols(self, dataset, columns):

        for c1,c2 in combinations(columns, 2): #permutations #Number of unique count where same i.e col1_col2 == col2_col1
        
            if (c1 == self.targetcol) | (c2 == self.targetcol):
                continue

            #From EDA to skip combination of ROLE_FAMILY and ROLE_CODE
            #if (c1 in ['ROLE_FAMILY', 'ROLE_CODE']) & (c2 in ['ROLE_FAMILY', 'ROLE_CODE']):
            #    continue
            
            name = "{}_{}".format(c1, c2)
            #new_cols.append(name)
            dataset[name] = dataset[c1] + " " + dataset[c2]

        return dataset

    def get_col_interactions_svd(self, dataset):

        new_dataset = pd.DataFrame()

        for col1, col2 in permutations(dataset.columns, 2):
        
            if (col1 == self.targetcol) | (col2 == self.targetcol):
                continue

            data = self.extract_col_interaction(dataset, col1, col2)

            col_name = [x for x in data.columns if "svd" in x] #[0]
            
            #Merge the extracted interaction data about col1 to main dataset by joining them with there key.
            #Reason to do this is we want to merge the calculated interaction data to respective col1 in main dataset
            new_dataset[col_name] = dataset[[col1]].merge(data, on = col1, how = 'left')[col_name]

        return new_dataset    

    def extract_col_interaction(self, dataset, col1, col2):

        key = col1 + "_" + col2
        
        #params = kwargs['params']
        data = dataset.groupby([col1])[col2].agg(lambda x: " ".join(x))

        if key in self.dict_Vectorizer:
            vectorizer = self.dict_Vectorizer[key]
        else:
            vectorizer = CountVectorizer(lowercase = False)            
            self.dict_Vectorizer[key] = vectorizer.fit(data) #Save the fitted vectorizer obj, which will be used in the transform stage
        
        data_X = vectorizer.transform(data)

        if key in self.dict_dim_reduction:
            dim_red = self.dict_dim_reduction[key] 
        else:
            dim_red = TruncatedSVD(n_components = self.params['dim_reduction'], random_state = self.params['random_seed'])
            self.dict_dim_reduction[key] = dim_red.fit(data_X) #Save the fitted dimension reduction obj, which will be used in the transform stage
        
        data_X = dim_red.transform(data_X)
        
        col_no = self.params['dim_reduction']
        col_names = []
        for i in range(col_no):
            col_names.append(col1 + "_{}_svd_{}_{}".format(col2, 'cv', i))
        data_X = pd.DataFrame( data_X, columns = col_names)
        
        #Output: Will be for each unique values of col1 will have calculated vectorizer of interacted col2
        result = pd.DataFrame()
        result[col1] = data.index.values
        result = pd.concat([result, data_X], axis = 1) 
        #result[col1 + "_{}_{}_svd".format(col2, 'cv' if feature_code == 'count_vector' else 'tf')] = data_X.ravel()

        return result

    def encode(self, X):

        col_use = [x for x in X.columns if not x in ['ROLE_TITLE', 'MGR_ID']]
        X = X[col_use]

        X = self.combine_cols(X, col_use)

        new_dataset = self.get_col_interactions_svd(X)

        return new_dataset

#==============================================================================================================

    def fit_transform(self, X, y = None):

        self.dict_Vectorizer = {}
        self.dict_dim_reduction = {}

        return self.encode(X)

    def transform(self, X, y = None):

        if (len(self.dict_Vectorizer) == 0) | (len(self.dict_dim_reduction) == 0):
            raise ModuleException('CountVect', 'Count Vectorizer instance is not fitted yet. Try calling fit_transform first.')
        
        return self.encode(X) 

        
