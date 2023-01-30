from src.models.feature_eng.CountVectorizerEncoding import CountVectorizerEncoding
from src.models.feature_eng.Combine_feature import CombineFeatures
from src.models.feature_eng.TE_KFold import KFoldTargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import utils.write_utils as hlpwrite
import utils.read_utils as hlpread
from os.path import exists, join

class employee_access_resource:

    def __init__(self, trained_model_path, feature_eng_object_path):

      self.model = None
      self.trained_model_path = trained_model_path
      if exists(trained_model_path) == True:
        self.model = hlpread.read_object(trained_model_path)


      self.feature_engg = None
      self.feature_eng_object_path = feature_eng_object_path
      if exists(feature_eng_object_path) == True:
        self.feature_engg = hlpread.read_object(feature_eng_object_path)

    def generate_feature(self, X):

        if self.feature_engg == None:
            self.feature_engg = Pipeline(steps = [
                                        ('combine_feature', CombineFeatures()),
                                        ('count_vectorizer_encoding', CountVectorizerEncoding()),
                                        ('KFoldTE', KFoldTargetEncoder())
                                    ]) 

            X = self.feature_engg.fit_transform(X) 

            #Save the feature eng 
            hlpwrite.save_object(self.feature_eng_object_path , self.feature_engg)            

        else:
            X = self.feature_engg.transform(X)
        
        return X

    def train(self, X):

        training_param =  hlpread.read_yaml_key('trained_model')

        #Define model
        self.model = DecisionTreeClassifier(criterion = 'gini', random_state = 42)
        self.model.set_params(**training_param['params'])

        #Generate features
        X = self.generate_feature(X)

        #Train model
        self.model.fit(X[X.columns[30:]], X['ACTION'])

        #Save the model
        hlpwrite.save_object(self.trained_model_path , self.model)

        return self

    def predict(self, X):

        if self.model == None:
            self.train(X)

        X = self.generate_feature(X)

        y_hat = self.model.predict_proba(X[X.columns[29:]]) #Predict will not have 'ACTION' FEATURE
        y_hat =  y_hat.argmax(-1)  
        
        return y_hat

if __name__ == '__main__':
    
    model_obj = employee_access_resource(hlpread.read_yaml_key('trained_model.model_path'), hlpread.read_yaml_key('trained_model.feature_eng'))

    train_data = join(hlpread.read_yaml_key('data_source.data_folders'),
                      hlpread.read_yaml_key('data_source.prepared.folder'),
                      hlpread.read_yaml_key('data_source.prepared.clean_train'),
                    )
    db_train = hlpread.read_from_parquet(train_data)

    model_obj.train(db_train)