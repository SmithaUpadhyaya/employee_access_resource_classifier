from src.models.feature_eng.TFIDFVectorizerEncoding import TFIDFVectorizerEncoding
from src.models.feature_eng.CountVectorizerEncoding import CountVectorizerEncoding
from src.models.feature_eng.ResourceEncodeByFeature import ResourceEncodeByFeature
from src.models.feature_eng.RandomCatagoryEncode import RandomCatagoryEncode
from src.models.feature_eng.KFoldFreqEncoding import KFoldFrequencyEncoding
from src.models.feature_eng.Combine_feature import CombineFeatures
from src.models.feature_eng.FreqEncoding import FrequencyEncoding
from src.models.feature_eng.TE_KFold import KFoldTargetEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import utils.write_utils as hlpwrite
import utils.read_utils as hlpread
from xgboost import XGBClassifier
from os.path import exists, join
import xgboost as xgb

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


    def generate_feature(self, X, pipeline_params):

        if self.feature_engg == None:

            self.feature_engg = Pipeline(steps = [
                                        ('combine_feature', CombineFeatures()), #This step is always required
                                    ]) 


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
        model = DecisionTreeClassifier(criterion = 'gini')
        #model = XGBClassifier(objective='binary:logistic')
        model.set_params(**params)

        return model

    def train(self, X):      
        

        self.model = self.define_model(self.training_param['params'])        

        #Generate features
        X.drop(['ROLE_TITLE', 'MGR_ID'], axis = 1, inplace = True)
        X = self.generate_feature(X, self.training_param['pipeline_type'])

        #Train model
        Y = X.ACTION
        X.drop('ACTION', axis = 1, inplace = True)
        feature_columns = X.select_dtypes(exclude = ['object']).columns #Exclude "object" type columns   

        self.model.fit(X[feature_columns], Y)

        #Save the model
        hlpwrite.save_object(self.trained_model_path , self.model)

        return self

    def predict(self, X):

        if self.model == None:
            self.train(X)

        X = self.generate_feature(X, self.training_param)

        if 'ACTION' in X.columns:
            X.drop('ACTION', axis = 1, inplace = True)

        if 'ROLE_TITLE' in X.columns:
            X.drop('ROLE_TITLE', axis = 1, inplace = True)

        if 'MGR_ID' in X.columns:
            X.drop('MGR_ID', axis = 1, inplace = True)

        feature_columns = X.select_dtypes(exclude = ['object']).columns

        y_hat = self.model.predict_proba(X[feature_columns]) #Predict will not have 'ACTION' FEATURE
        #y_hat =  y_hat.argmax(-1)  
        
        return y_hat

if __name__ == '__main__':
    
    model_obj = employee_access_resource(hlpread.read_yaml_key('trained_model.model_path'), hlpread.read_yaml_key('trained_model.feature_eng'))

    train_data = join(hlpread.read_yaml_key('data_source.data_folders'),
                      hlpread.read_yaml_key('data_source.prepared.folder'),
                      hlpread.read_yaml_key('data_source.prepared.clean_train'),
                    )
    db_train = hlpread.read_from_parquet(train_data)

    model_obj.train(db_train)