from src.models.feature_eng.TFIDFVectorizerEncoding import TFIDFVectorizerEncoding
from src.models.feature_eng.CountVectorizerEncoding import CountVectorizerEncoding
from src.models.feature_eng.ResourceEncodeByFeature import ResourceEncodeByFeature
from src.models.feature_eng.RandomCatagoryEncode import RandomCatagoryEncode
from src.models.feature_eng.KFoldFreqEncoding import KFoldFrequencyEncoding
from src.models.feature_eng.Combine_feature import CombineFeatures
from src.models.feature_eng.FreqEncoding import FrequencyEncoding
from src.models.feature_eng.TE_KFold import KFoldTargetEncoder
#from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from category_encoders import BinaryEncoder
from sklearn.pipeline import Pipeline
import utils.write_utils as hlpwrite
import utils.read_utils as hlpread
#from xgboost import XGBClassifier
from os.path import exists, join
import pandas as pd
#import xgboost as xgb

class employee_access_resource:

    def __init__(self, trained_model_path, feature_eng_object_path):

      self.model = None
      self.feature_engg = None
      self.training_param =  hlpread.read_yaml_key('trained_model')

      self.trained_model_path = trained_model_path
      if exists(trained_model_path) == True:
        self.model = hlpread.read_object(trained_model_path)
   
      self.feature_eng_object_path = feature_eng_object_path
      if exists(feature_eng_object_path) == True:
        self.feature_engg = hlpread.read_object(feature_eng_object_path)

      self.threshold = 0.9364837895388629


    def generate_feature(self, X, pipeline_params):

        if self.feature_engg == None:

            self.feature_engg = Pipeline(steps = [
                                        #('combine_feature', CombineFeatures()), #This step is always required
                                    ]) 

            if pipeline_params['combine_feature'] == True:
                self.feature_engg.steps.append(('combine_feature', CombineFeatures()))

            if pipeline_params['tfidf_vectorizer_encoding'] == True:
                self.feature_engg.steps.append(('tfidf_vectorizer_encoding', TFIDFVectorizerEncoding()))

            if pipeline_params['count_vectorizer_encoding'] == True:
                self.feature_engg.steps.append(('count_vectorizer_encoding', CountVectorizerEncoding()))

            if pipeline_params['KFoldTE'] == True:
                self.feature_engg.steps.append(('KFoldTE', KFoldTargetEncoder()))

            if pipeline_params['frequency_encoding'] == True:
                self.feature_engg.steps.append(('frequency_encoding', FrequencyEncoding()))

            if pipeline_params['KFold_frequency_encoding'] == True:
                self.feature_engg.steps.append(('KFold_frequency_encoding', KFoldFrequencyEncoding()))

            if pipeline_params['random_catagory_encode'] == True:
                self.feature_engg.steps.append(('random_catagory_encode', RandomCatagoryEncode()))

            if pipeline_params['resource_catagory_encode'] == True:
                self.feature_engg.steps.append(("encode_resource_grpby_role_deptname_role_family", ResourceEncodeByFeature()))
                #self.feature_engg.steps.append(("encode_resource_grpby_role_deptname_role_family", FunctionTransformer(encode_resource_by_feature)))

            if pipeline_params['binary_encode'] == True:
                self.feature_engg.steps.append(("binary_encoder", BinaryEncoder(cols = hlpread.read_yaml_key('featurize.binary_encoder.columns') , base = 2)))

            #self.feature_engg = Pipeline(steps = [
            #                            ('combine_feature', CombineFeatures()),
            #                            #('tfidf_vectorizer_encoding', TFIDFVectorizerEncoding()),
            #                            #('count_vectorizer_encoding', CountVectorizerEncoding()), 
            #                            #('KFoldTE', KFoldTargetEncoder()), 
            #                            #('KFold_frequency_encoding', KFoldFrequencyEncoding()),                                      
            #                            ('frequency_encoding', FrequencyEncoding()),
            #                            #('random_catagory_encode', RandomCatagoryEncode()),
            #                        ]) 

            X = self.feature_engg.fit_transform(X) 

            #Save the feature eng 
            hlpwrite.save_object(self.feature_eng_object_path , self.feature_engg)            

        else:
            X = self.feature_engg.transform(X)
        
        return X

    def define_model(self, params):

        #Define model
        model = LogisticRegression()
        #model = DecisionTreeClassifier(criterion = 'gini')
        #model = XGBClassifier(objective='binary:logistic')
        model.set_params(**params)
        """
        base_model = DecisionTreeClassifier()
        base_model.set_params(**params['base_estimator'])   

        bagg_params = params['bagging']
        model = BaggingClassifier(estimator = base_model,
                                  n_estimators = bagg_params['n_estimators'], #Lets keep the it same as we have define for cv
                                  max_samples = 1.0 - bagg_params['test_size'], 
                                  bootstrap = True,
                                  random_state = bagg_params['random_seed']
                                )
        """
        return model

    def train(self, X):      
        
        self.model = self.define_model(self.training_param['params'])        

        #Generate features
        X.drop(['ROLE_TITLE', 'MGR_ID'], axis = 1, inplace = True)

        #Train model. Moved it for BinaryEncoder Pipeline
        Y = X.ACTION
        X.drop('ACTION', axis = 1, inplace = True)

        X = self.generate_feature(X, self.training_param['pipeline_type'])
                
        feature_columns = X.select_dtypes(exclude = ['object']).columns #Exclude "object" type columns   

        self.model.fit(X[feature_columns], Y)

        #Save the model
        hlpwrite.save_object(self.trained_model_path , self.model)

        return self

    def predict_score(self, X):

        if self.model == None:
            self.train(X)

        if 'ACTION' in X.columns:
            X.drop('ACTION', axis = 1, inplace = True)

        if 'ROLE_TITLE' in X.columns:
            X.drop('ROLE_TITLE', axis = 1, inplace = True)

        if 'MGR_ID' in X.columns:
            X.drop('MGR_ID', axis = 1, inplace = True)

        X = self.generate_feature(X, self.training_param)

        feature_columns = X.select_dtypes(exclude = ['object']).columns

        y_hat = self.model.predict_proba(X[feature_columns]) #Predict will not have 'ACTION' FEATURE
        #y_hat =  y_hat.argmax(-1)  
        
        return y_hat

    def predict(self, X):

        score = self.predict_score(X)[0][1]

        if score < self.threshold:
            return score, 0
        else:
            return score, 1

    def check_access(self, 
                     resource: str, 
                     role_rollup_1: str, role_rollup_2: str, 
                     role_family: str, role_family_desc: str, 
                     role_deptname: str, role_code: str
                    ):

        
        X = pd.DataFrame( [[resource, role_rollup_1, role_rollup_2, role_deptname, role_family_desc, role_family, role_code]], 
                         columns = ['RESOURCE', 
                                    'ROLE_ROLLUP_1', 
                                    'ROLE_ROLLUP_2', 
                                    'ROLE_DEPTNAME', 
                                    'ROLE_FAMILY_DESC',	
                                    'ROLE_FAMILY',	
                                    'ROLE_CODE']
                        )
        return self.predict(X)

if __name__ == '__main__':
    
    model_obj = employee_access_resource(hlpread.read_yaml_key('trained_model.model_path'), hlpread.read_yaml_key('trained_model.feature_eng'))

    train_data = join(hlpread.read_yaml_key('data_source.data_folders'),
                      hlpread.read_yaml_key('data_source.prepared.folder'),
                      hlpread.read_yaml_key('data_source.prepared.clean_train'),
                    )
    db_train = hlpread.read_from_parquet(train_data)

    model_obj.train(db_train)