import os
import pickle
import logs.logger as log
import utils.read_utils as hlpread
import utils.write_utils as hlpwrite
from sklearn.pipeline import Pipeline
#from dvclive.keras import DVCLiveCallback #This will work with keras library and not with model sklearn. Since this require to define Callback 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def define_model(param_filepath):

    model_param = hlpread.read_yaml_key('model', param_filepath)
    model_type = model_param['model_type']

    log.write_log(f'train_model: Train model type: \"{model_type}\"...', log.logging.DEBUG)
    hyper_param = model_param[model_type]['hyper_params']

    if model_type == 'logistic_reg':

        model = LogisticRegression(random_state = 42)
        model.set_params(**hyper_param)

    elif model_type == 'decision_tree':
        
        model = DecisionTreeClassifier(criterion = 'gini', random_state = 42)
        model.set_params(**hyper_param)   
        
    else:
        raise Exception('Unsupported model_type.')
        
    return model, model_param


if __name__ == '__main__':

    log.write_log(f'train_model: Model training started.', log.logging.DEBUG)

    train_params_file = None
    #train_params_file = os.path.join("src", "data", "train_params.yaml")

    #Step 1: Load training data    
    #train_params = hlpread.read_yaml_key('training_data', train_params_file)    
    #train_filepath = os.path.join(
    #                                hlpread.read_yaml_key('data_source.data_folders'),
    #                                train_params['output']['folder'],
    #                                train_params['output']['filename'],                                  
    #                            )
    train_filepath = os.path.join(
                                    hlpread.read_yaml_key('data_source.data_folders'),
                                    hlpread.read_yaml_key('train_test_split.train_data', train_params_file)                                   
                                )

    log.write_log(f'train_model: Loading training data from \"{train_filepath}\" started...', log.logging.DEBUG)
    X_train = hlpread.read_from_parquet(train_filepath)
    Y_train = X_train['ACTION']
    X_train.drop(columns = ['ACTION'], inplace = True)
    log.write_log(f'train_model: Total number of records to training: {X_train.shape}.', log.logging.DEBUG)

    #Step 2: Define the model to train
    log.write_log(f'train_model: Define the model to train...', log.logging.DEBUG)
    model, model_param = define_model(train_params_file)

    #For LogReg model we need to standarize the freq_cnt feature as can range from 0 to +ve inf
    if model_param['model_type'] == 'logistic_reg':     
        
        freq_enc_cols = [x for x in X_train.columns if 'FreqEnc'.lower() in x.lower()]
        
        if len(freq_enc_cols) != 0:

            log.write_log(f'train_model: Adding standard scaler transformer for freq enc feature: {freq_enc_cols}...', log.logging.DEBUG)

            # ColumnTransformer which applies transformers to a specified set of columns of an array or pandas DataFrame
            preprocessor = ColumnTransformer(transformers = [("standard_scaler_freq_enc", StandardScaler(), freq_enc_cols)])

            #Pipeline to perform standarization of freq enc feature before fitting the model
            model = Pipeline(steps=[("preprocessor", preprocessor), ("logreg_model", model)] )
    
    
    #Step 3: Fit Model
    log.write_log(f'train_model: Fit model started...', log.logging.DEBUG)
    model.fit(X_train, 
              Y_train, 
              #callbacks = [DVCLiveCallback(save_dvc_exp=True)]
              )
    log.write_log(f'train_model: Fit model completed...', log.logging.DEBUG)

    #Step 4: Save trained model to pickel file.
    log.write_log(f'train_model: Save trained mode to pickel file.', log.logging.DEBUG)
    save_model_path = os.path.join(
                                    hlpread.read_yaml_key('data_source.data_folders'),
                                    model_param['trained_model']                               
                                  )
    hlpwrite.save_object(save_model_path, model)
    #with open(save_model_path, "wb") as fd:
    #    pickle.dump(model, fd)


    log.write_log(f'train_model: Model training completed.', log.logging.DEBUG)


