import os
import utils.read_utils as hlpread
import utils.write_utils as hlpwrite
from sklearn.pipeline import Pipeline
from src.models.feature_eng.TE_KFold import KFoldTargetEncoder
from src.models.feature_eng.FreqEncoding import FrequencyEncoding
from src.models.feature_eng.Combine_feature import CombineFeatures
from src.models.feature_eng.KFoldFreqEncoding import KFoldFrequencyEncoding
from src.models.feature_eng.CountVectorizerEncoding import CountVectorizerEncoding
from src.models.feature_eng.TFIDFVectorizerEncoding import TFIDFVectorizerEncoding


#Load Cleaned data 
clean_train_data = os.path.join(hlpread.read_yaml_key('data_source.data_folders'),
                                hlpread.read_yaml_key('data_source.prepared.folder'),
                                hlpread.read_yaml_key('data_source.prepared.clean_train'),
                                )
db_train = hlpread.read_from_parquet(clean_train_data)

params = hlpread.read_yaml_key('data_source.feature')

feature_engg = Pipeline( steps = [
                                ('combine_feature', CombineFeatures()),
                                ('tfidf_vectorizer_encoding', TFIDFVectorizerEncoding()),
                                ('KFoldTE', KFoldTargetEncoder()),
                                ])

X = feature_engg.fit_transform(db_train) 

#Drop all the orginal feature after transform
X.drop(columns = X.columns[29:], inplace = True)


save_file_path = os.path.join(
                                hlpread.read_yaml_key('data_source.data_folders'),
                                params['output']['folder'],
                                params['output']['filename'],                                  
                                )

hlpwrite.save_to_parquet(X, save_file_path, True)