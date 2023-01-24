import gc
import os
import pandas as pd
import logs.logger as log
import utils.read_utils as hlpread
import utils.write_utils as hlpwrite
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from src.models.feature_eng.TE_KFold import KFoldTargetEncoder
from src.models.feature_eng.FreqEncoding import FrequencyEncoding
from src.models.feature_eng.Combine_feature import CombineFeatures
from src.models.feature_eng.KFoldFreqEncoding import KFoldFrequencyEncoding
from src.models.feature_eng.CountVectorizerEncoding import CountVectorizerEncoding
from src.models.feature_eng.TFIDFVectorizerEncoding import TFIDFVectorizerEncoding

if __name__ == '__main__':

    train_params_file = os.path.join("src", "data", "train_params.yaml")

    #Step 1: Load Cleaned data 
    log.write_log(f'generate_training_features: Loading clean data started...', log.logging.DEBUG)
    clean_train_data = os.path.join(hlpread.read_yaml_key('data_source.data_folders'),
                                    hlpread.read_yaml_key('data_source.prepared.folder'),
                                    hlpread.read_yaml_key('data_source.prepared.clean_train'),
                                    )
    db_train = hlpread.read_from_parquet(clean_train_data)
    
    

    #Step 2: Generate features pipeline
        
    #Load the pipeline configuration
    log.write_log(f'generate_training_features: Load pipeline configuartion...', log.logging.DEBUG)
    pipeline_params = hlpread.read_yaml_key('pipeline_type', train_params_file)

    
    log.write_log(f'generate_training_features: Define the pipeline...', log.logging.DEBUG)
    feature_engg = Pipeline(steps = [
                                        ('combine_feature', CombineFeatures()), #This step is always required
                                    ]) 

    #if pipeline_params['combine_feature'] == True:
    #    feature_engg.steps.append(['combine_feature', CombineFeatures()])

    if pipeline_params['KFoldTE'] == True:
        feature_engg.steps.append(('KFoldTE', KFoldTargetEncoder()))

    if pipeline_params['frequency_encoding'] == True:
        feature_engg.steps.append(('frequency_encoding', FrequencyEncoding(min_group_size = 1)))

    if pipeline_params['KFold_frequency_encoding'] == True:
        feature_engg.steps.append(('KFold_frequency_encoding', KFoldFrequencyEncoding(min_group_size = 1)))

    if pipeline_params['tfidf_vectorizer_encoding'] == True:
        feature_engg.steps.append(('tfidf_vectorizer_encoding', TFIDFVectorizerEncoding()))

    if pipeline_params['count_vectorizer_encoding'] == True:
        feature_engg.steps.append(('count_vectorizer_encoding', CountVectorizerEncoding()))

    log.write_log(f'generate_training_features: Fit_transform pipeline started...', log.logging.DEBUG)
    X = feature_engg.fit_transform(db_train) 
    log.write_log(f'generate_training_features: Fit_transform pipeline completed...', log.logging.DEBUG)

    #Drop all the orginal feature after transform
    X.drop(columns = X.columns[29:], inplace = True)

    Y = X[['ACTION']]
    X.drop(columns = ['ACTION'], inplace = True)


    #Split train and test data
    log.write_log(f'generate_training_features: Split train and test data...', log.logging.DEBUG)

    #from sklearn.model_selection import train_test_split
    #X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.02, shuffle = True, random_state = 2023)

    #Stratified Split that preserve +ve and -ve class in the train and test.
    #Kept small tarin size and we have few -ve sample and we want the model to train for -ve sample
    split_params = hlpread.read_yaml_key('train_test_split', train_params_file)
    split_s = StratifiedShuffleSplit(n_splits = 1, test_size = split_params['test_size'], random_state = split_params['random_seed'])

    for train_index, test_index in split_s.split(X, Y):
        
        Y_train, Y_test = Y.iloc[train_index,:], Y.iloc[test_index,:]
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

    del [X,Y]
    gc.collect()

    X_train = pd.concat([X_train, Y_train], axis = 1 )
    X_test = pd.concat([X_test, Y_test], axis = 1 )

    log.write_log(f'generate_training_features: Split train and test data completed...', log.logging.DEBUG)

    #Save train data
    log.write_log(f'generate_training_features: Save train and test split data to files...', log.logging.DEBUG)
    
    train_data_param = hlpread.read_yaml_key('training_data', train_params_file)
    train_filepath = os.path.join(
                                    hlpread.read_yaml_key('data_source.data_folders'),
                                    train_data_param['output']['folder'],
                                    train_data_param['output']['filename'],                                  
                                )

    hlpwrite.save_to_parquet(X_train, train_filepath, True)


    #Save test data
    test_data_param = hlpread.read_yaml_key('test_data', train_params_file)
    test_filepath = os.path.join(
                                    hlpread.read_yaml_key('data_source.data_folders'),
                                    test_data_param['output']['folder'],
                                    test_data_param['output']['filename'],                                  
                                )
    hlpwrite.save_to_parquet(X_test, test_filepath, True)

    log.write_log(f'generate_training_features: Generated features for training...', log.logging.DEBUG)
