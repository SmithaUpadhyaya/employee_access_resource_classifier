import os
import json
import pickle
import numpy as np
import logs.logger as log
from xgboost import XGBClassifier
import utils.read_utils as hlpread
import utils.write_utils as hlpwrite
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
#from dvclive.keras import DVCLiveCallback #This will work with keras library and not with model sklearn. Since this require to define Callback 
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedShuffleSplit


def define_model(param_filepath):

    model_param = hlpread.read_yaml_key('model', param_filepath)
    model_type = model_param['model_type']

    log.write_log(f'train_model: Train model type: \"{model_type}\"...', log.logging.DEBUG)
    hyper_param = model_param[model_type]['hyper_params']

    if model_type == 'logistic_reg':

        model = LogisticRegression()
        model.set_params(**hyper_param)

    elif model_type == 'decision_tree':
        
        model = DecisionTreeClassifier(criterion = 'gini')
        model.set_params(**hyper_param)   

    elif model_type == 'extra_decision_tree':

        model = ExtraTreesClassifier(criterion = 'gini')
        model.set_params(**hyper_param)

    elif model_type == 'random_forest':

        model = RandomForestClassifier(criterion='gini')
        model.set_params(**hyper_param)

    elif model_type == 'xgboost':

        model = XGBClassifier(objective='binary:logistic')
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
    X = hlpread.read_from_parquet(train_filepath)
    log.write_log(f'train_model: Total number of records to training: {X.shape}.', log.logging.DEBUG)

    #Step 2: Define the model to train
    log.write_log(f'train_model: Define the model to train...', log.logging.DEBUG)
    model, model_param = define_model(train_params_file)

    #For LogReg model we need to standarize the freq_cnt feature as can range from 0 to +ve inf
    if model_param['model_type'] == 'logistic_reg':     
        
        freq_enc_cols = [x for x in X.columns if ('FreqEnc'.lower() in x.lower()) | ('cv'.lower() in x.lower())]
        
        if len(freq_enc_cols) != 0:

            log.write_log(f'train_model: Adding standard scaler transformer for freq enc feature: {freq_enc_cols}...', log.logging.DEBUG)

            # ColumnTransformer which applies transformers to a specified set of columns of an array or pandas DataFrame
            preprocessor = ColumnTransformer(transformers = [("standard_scaler_freq_enc", StandardScaler(), freq_enc_cols)])

            #Pipeline to perform standarization of freq enc feature before fitting the model
            model = Pipeline(steps=[("preprocessor", preprocessor), ("logreg_model", model)] )

    #elif model_param['model_type'] == 'decision_tree':

    #    freq_enc_cols = ['ROLE_DEPTNAME', 'ROLE_ROLLUP_1_ROLE_DEPTNAME', 'ROLE_ROLLUP_2_ROLE_DEPTNAME', 'ROLE_ROLLUP_2_ROLE_CODE', 'ROLE_DEPTNAME_ROLE_FAMILY_DESC', 'ROLE_DEPTNAME_ROLE_FAMILY', 'ROLE_DEPTNAME_ROLE_CODE']
    #    for i in range(len(freq_enc_cols)):
    #        freq_enc_cols[i] = freq_enc_cols[i] + '_FreqEnc'

    #    preprocessor = ColumnTransformer(transformers = [("log_transform_freq_enc_log", FunctionTransformer(np.log10), freq_enc_cols)])
    #    model = Pipeline(steps=[("preprocessor", preprocessor), ("decision_tree_model", model)] )
        
    #Step 3: Fit Model
    log.write_log(f'train_model: Fit model started...', log.logging.DEBUG)

    cv_roc = {}
    cv_f1_score = {}

    split_params = hlpread.read_yaml_key('train_test_split', train_params_file)
    split_s = StratifiedShuffleSplit(n_splits = split_params['cv'], test_size = split_params['test_size'], random_state = split_params['random_seed'])

    fold = 0
    for train_index, test_index in split_s.split(X, X.ACTION):
           
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

        Y_train = X_train.ACTION
        Y_test = X_test.ACTION
        #Y_train, Y_test = X[['ACTION']].iloc[train_index,:], X[['ACTION']].iloc[test_index,:]
        
        X_train.drop('ACTION', axis = 1, inplace = True)
        X_test.drop('ACTION', axis = 1, inplace = True)

        model.fit(X_train, Y_train ) 

        Y_test_pred = model.predict_proba(X_test).astype(float)
        auc_score = roc_auc_score(Y_test.astype(float), Y_test_pred[:,1])
        cv_roc[fold] = auc_score

        
        cv_f1_score[fold] = f1_score(Y_test.astype(float), Y_test_pred.argmax(-1) )
        
        fold += 1

    #model.fit(X_train, 
    #          Y_train, 
    #          #callbacks = [DVCLiveCallback(save_dvc_exp=True)]
    #          )
    
    log.write_log(f'train_model: Fit model completed...', log.logging.DEBUG)

    #Step 4:
    #Save the model score over cv

    metrics = {
        'roc_auc': sum(cv_roc.values()) / len(cv_roc),
        'f1_score': sum(cv_f1_score.values()) / len(cv_f1_score),

    }
    metric_file = os.path.join(
                                hlpread.read_yaml_key('data_source.data_folders'),                                
                                model_param[model_param['model_type']]['eval_metrics']
                             )
    os.makedirs(metric_file, exist_ok = True)

    json.dump(
        obj = metrics,
        fp = open(os.path.join(metric_file, "metrics.json"), 'w'),
        indent = 4, 
        sort_keys = True
    )

    #Save roc and f1 score for each fold in seperate file
    json.dump(
        {
            "f1_score": cv_f1_score,
            "roc_score": cv_roc
        },
        fp = open(os.path.join(metric_file, "folds_scores.json"), 'w'),
        indent = 4, 
        sort_keys = True
    )

    #Metric file for dvc exp output
    metric_file = os.path.join( hlpread.read_yaml_key('data_source.data_folders'), 
                                "model","metrics")
    os.makedirs(metric_file, exist_ok = True)

    json.dump(
        obj = metrics,
        fp = open(os.path.join(metric_file, "metrics.json"), 'w'),
        indent = 4, 
        sort_keys = True
    )
    
    """
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
    """

