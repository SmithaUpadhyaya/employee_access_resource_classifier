import os
import json
from dvclive import Live 
import logs.logger as log
import matplotlib.pyplot as plt
import utils.read_utils as hlpread
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

train_params_file = os.path.join("src", "data", "train_params.yaml")

def eval(Y, Y_predictions_by_class):

    #Folder where the eval metric will be store
    eval_metric = hlpread.read_yaml_key('eval', train_params_file)
    dvc_live_path = os.path.join(hlpread.read_yaml_key('data_source.data_folders'),
                                 eval_metric['evals'])
    os.makedirs(dvc_live_path, exist_ok = True)
    dvc_live = Live(dvc_live_path)


    Y_hat = Y_predictions_by_class[:,1] #Y_hat return 2d array where it provide probality of item belong to either of two class. First col is for "0" lable and Second col is for "1"
    
    #Stored the summary value of the eval. 
    summary = {} #Init to empty dict

    #Precision and Recall Curve
    pr_auc_score = average_precision_score(Y, Y_hat)
    summary["pr_auc"] = pr_auc_score 
    dvc_live.log_sklearn_plot("precision_recall", Y, Y_hat, name = 'Precision Recall Curve')
    
    #ROC
    auc_score = roc_auc_score(Y, Y_hat)
    summary["roc_auc"] = auc_score
    dvc_live.log_sklearn_plot("roc", Y, Y_hat, name = "ROC")

    #Confusion_matric
    dvc_live.log_sklearn_plot("confusion_matrix", Y, Y_hat, name = "Confusion Matrix")
    
    #Compute False and True Positive rate from confusion matrix
    Y_pred = Y_predictions_by_class.argmax(-1)  
    conf_matrix = confusion_matrix(Y, Y_pred)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if (i == 0) & (j == 0):
                tn = conf_matrix[i, j]
            elif (i == 0) & (j == 1):
                fp = conf_matrix[i, j]
            elif (i == 1) & (j == 0):
                fn = conf_matrix[i, j]
            else:
                tp = conf_matrix[i, j]
    #False Positive Rate    
    summary["fpr"] = fp / (fp + tn)
    #True Positive Rate
    summary["tpr"] = tp / (fn + tp)
    
    #To save the summary updated to metric.json file in the folder define when init dvcLive()
    metric_file = os.path.join(dvc_live_path, eval_metric['eval_metrics'],)
    os.makedirs(metric_file, exist_ok = True)
    metric_file = os.path.join(metric_file, "metrics.json")
    json.dump(
        obj = summary,
        fp = open(metric_file, 'w'),
        indent = 4, 
        sort_keys = True
    )

    """Old code to manully created the eval json file
    #Folder where the eval metric will be store
    eval_metric = hlpread.read_yaml_key('eval', train_params_file)

    Y_hat = Y_predictions_by_class[:,1] #Y_hat return 2d array where it provide probality of item belong to either of two class. First col is for "0" lable and Second col is for "1"
    
    #pr_rc curve
    pr_auc_score = average_precision_score(Y, Y_hat)
    precision, recall, prc_thresholds = precision_recall_curve(Y, Y_hat)
    
    prc_file = os.path.join(
                            hlpread.read_yaml_key('data_source.data_folders'),
                            eval_metric['evals'],
                            eval_metric['eval_plots'],
                            'pr_rc' #To keep reproducibility, outputs should be in separate tracked directories 
                            )
    os.makedirs(prc_file, exist_ok = True)    
    prc_file = os.path.join(prc_file, "pr_rc_curve.json")
    
    with open(prc_file, "w") as fd:
        json.dump(
            [
                #"prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in zip(precision, recall, prc_thresholds)
                #]
            ],
            fd,
            indent = 4,
        )

    #roc curve
    auc_score = roc_auc_score(Y, Y_hat)
    fpr, tpr, thresholds = roc_curve(Y, Y_hat)
    
    roc_file = os.path.join(
                            hlpread.read_yaml_key('data_source.data_folders'),
                            eval_metric['evals'],
                            eval_metric['eval_plots'],
                            'roc' #To keep reproducibility, outputs should be in separate tracked directories 
                           )
    os.makedirs(roc_file, exist_ok = True)
    roc_file = os.path.join(roc_file, "roc_curve.json")
 
    with open(roc_file, "w") as fd:
        json.dump(
            [
                #"roc": [
                        {"fpr": fp, "tpr": tp, "threshold": t}
                        for fp, tp, t in zip(fpr, tpr, thresholds)
                #]
            ],
            fd,
            indent = 4,
        )

    #Plot confusion_matric
    # This will return index of max value. 
    # In our case it will return column index which has max value in a row.

    confusion_file = os.path.join(
                                  hlpread.read_yaml_key('data_source.data_folders'),
                                  eval_metric['evals'],
                                  eval_metric['eval_plots'],
                                  'cm' #Confusion matrix folder #For DVC To keep reproducibility, outputs should be in separate tracked directories all
                           )
    os.makedirs(confusion_file, exist_ok = True)
    confusion_matx_file = os.path.join(confusion_file, "confusion_matrix.png")

    #Compute matrix
    Y_pred = Y_predictions_by_class.argmax(-1)  
    conf_matrix = confusion_matrix(Y, Y_pred)

    #Plot
    fig, ax = plt.subplots(figsize = (7.5, 7.5))
    ax.matshow(conf_matrix, cmap = plt.cm.Blues, alpha=0.3)
    
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x = j, y = i,s = conf_matrix[i, j], va = 'center', ha = 'center', size = 'xx-large')

            if (i == 0) & (j == 0):
                tn = conf_matrix[i, j]
            elif (i == 0) & (j == 1):
                fp = conf_matrix[i, j]
            elif (i == 1) & (j == 0):
                fn = conf_matrix[i, j]
            else:
                tp = conf_matrix[i, j]

            

    plt.ylabel('Actuals', fontsize = 8)
    plt.xlabel('Predictions', fontsize = 8)
    plt.title(f'Confusion Matrix', fontsize = 8)
    plt.savefig(confusion_matx_file)

    #False Positive Rate    
    fpr = fp / (fp + tn)

    #True Positive Rate
    tpr = tp / (fn + tp)
    
    #This become our summary on model metrics.json
    metrics = {
        #'f1_score': f1_Score,
        'roc_auc': auc_score,
        'pr_auc_score': pr_auc_score,
        'tpr': tpr,
        'fpr': fpr
    }
    metric_file = os.path.join(
                                hlpread.read_yaml_key('data_source.data_folders'),
                                eval_metric['evals'],
                                eval_metric['eval_metrics'],
                             )
    os.makedirs(metric_file, exist_ok = True)
    metric_file = os.path.join(metric_file, "metrics.json")
    json.dump(
        obj = metrics,
        fp = open(metric_file, 'w'),
        indent = 4, 
        sort_keys = True
    )
    """
    
    
    
    
   

if __name__ == '__main__':

    log.write_log(f'train_model: Evaluate trained model started.', log.logging.DEBUG)
    eval_param = hlpread.read_yaml_key('eval', train_params_file)

    #Step 1: Load trained model
    log.write_log(f'train_model: Load trained model...', log.logging.DEBUG)
    trained_model = hlpread.read_object(os.path.join(hlpread.read_yaml_key('data_source.data_folders'),
                                                     hlpread.read_yaml_key('model.trained_model', train_params_file)
                                                    )
                                        )
    
    
    #Step 2: Load the training data
    log.write_log(f'train_model: Loading training data from started...', log.logging.DEBUG)
    X = hlpread.read_from_parquet(os.path.join(
                                              hlpread.read_yaml_key('data_source.data_folders'),
                                              hlpread.read_yaml_key('training_data.output.folder', train_params_file),
                                              hlpread.read_yaml_key('training_data.output.filename', train_params_file)                                                                             
                                             ))
    Y = X['ACTION']
    X.drop(columns = ['ACTION'], inplace = True)

    #Step 3: Predict 
    Y_predictions_by_class = trained_model.predict_proba(X).astype(float) #Return 2d numpy array which is the probaility for each class label
    Y = Y.astype(float)

    #Step 4: Generate Eval metric
    eval(Y, Y_predictions_by_class)
    

    