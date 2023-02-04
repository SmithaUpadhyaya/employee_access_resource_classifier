from utils.paths import get_project_root
from tqdm import tqdm
import numpy as np
import subprocess
import itertools
import random
import os


# Automated random search experiments
num_exps = 30 #Number of experiments to run to generate
random.seed(42)


"""
i = 3 #Since in my project structure are max at 3 level
wrk_dir = os.getcwd()
while i > 0:  
    if not os.path.exists(os.path.join(wrk_dir, 'params.yaml')):       
        wrk_dir = os.path.abspath(os.path.join(wrk_dir, os.pardir))        
    i = i - 1

print(f'Working directory: {wrk_dir}')
"""
def run_exp_Decision_Tree():

    #Hyper-paramters tuning for Decision Tree
    
    params = {
        "random_state": random.randint(50, 3000), 
        "max_depth": random.choice(range(3, 5, 1)),      #random.choice([3,5,7,9])
        "splitter": 'best', #random.choice(['best', 'random']),
        "min_samples_leaf":  random.choice([0.01, 0.05, 0.001, 0.002, 0.005]),
        "max_features":  random.choice(['sqrt', 'log2', 0.3, 0.5, 0.6, 0.7, 0.8, 0.95, 1]),
        #"min_samples_split":  random.choice([]),

        #Select the featurization techinique
        "KFoldTE": False, #random.choice([True, False]), 
        "frequency_encoding": False, #random.choice([True, False]),
        "KFold_frequency_encoding": True, #random.choice([True, False])
        "tfidf_vectorizer_encoding": False, #random.choice([True, False]),
        "count_vectorizer_encoding": True, #random.choice([True, False]),
    }


    subprocess.run(["dvc", "exp", "run", #"--queue",
                    "--set-param", f"model.decision_tree.hyper_params.random_state={params['random_state']}",
                    "--set-param", f"model.decision_tree.hyper_params.max_depth={params['max_depth']}",
                    "--set-param", f"model.decision_tree.hyper_params.splitter={params['splitter']}",
                    "--set-param", f"model.decision_tree.hyper_params.min_samples_leaf={params['min_samples_leaf']}",
                    "--set-param", f"model.decision_tree.hyper_params.max_features={params['max_features']}",
                  

                    #Select the featurization techinique
                    "--set-param", f"model.decision_tree.pipeline_type.KFoldTE={params['KFoldTE']}",
                    "--set-param", f"model.decision_tree.pipeline_type.frequency_encoding={params['frequency_encoding']}",
                    "--set-param", f"model.decision_tree.pipeline_type.KFold_frequency_encoding={params['KFold_frequency_encoding']}",
                    "--set-param", f"model.decision_tree.pipeline_type.tfidf_vectorizer_encoding={params['tfidf_vectorizer_encoding']}",
                    "--set-param", f"model.decision_tree.pipeline_type.count_vectorizer_encoding={params['count_vectorizer_encoding']}",

                    ]
                  #This did not help
                  #, cwd = get_project_root()
                  #, cwd = wrk_dir 
                  )

#===============================================================================================

def run_exp_ExtraTreesClassifier():
    
    #Hyperparamater tuning for ExtraTreesClassifier
    params = {
        "n_estimators": random.choice(range(50, 100, 5)),#random.choice([5, 10, 15, 20, 25]),
        "max_depth": random.choice([4, 5]), #random.choice([5,6]),

        "bootstrap": True, #random.choice([True, False]),
        "max_samples": random.choice(np.arange(0.65, 1, 0.1)), #random.choice([0.01, 0.3, 0.5, 0.6, 0.7, 0.8, 0.95, 1]),
        
        "max_features":  random.choice(np.arange(0.5, 1 , 0.1)),#random.choice([0.3, 0.5, 0.6, 0.7, 0.8, 0.95, 1]),
        "min_samples_leaf":  random.choice(np.arange(0.5, 0.9 , 0.1)), #random.choice([0.01, 0.05, 0.001, 0.002, 0.005]),

        "class_weight": random.choice(['balanced', 'balanced_subsample']),

        #Select the featurization techinique
        "KFoldTE": random.choice([True, False]), 
        "frequency_encoding": random.choice([True, False]),
        "KFold_frequency_encoding": random.choice([True, False]),
        "tfidf_vectorizer_encoding": random.choice([True, False]),
        "count_vectorizer_encoding": random.choice([True, False]),
    }

    subprocess.run(["dvc", "exp", "run", 
                    "--set-param", f"model.extra_decision_tree.hyper_params.n_estimators={params['n_estimators']}",
                    "--set-param", f"model.extra_decision_tree.hyper_params.max_depth={params['max_depth']}",

                    "--set-param", f"model.extra_decision_tree.hyper_params.bootstrap={params['bootstrap']}",
                    "--set-param", f"model.extra_decision_tree.hyper_params.max_samples={params['max_samples']}",

                    "--set-param", f"model.extra_decision_tree.hyper_params.max_features={params['max_features']}",
                    "--set-param", f"model.extra_decision_tree.hyper_params.min_samples_leaf={params['min_samples_leaf']}",

                    "--set-param", f"model.extra_decision_tree.hyper_params.class_weight={params['class_weight']}",


                    #Select the featurization techinique
                    "--set-param", f"model.extra_decision_tree.pipeline_type.KFoldTE={params['KFoldTE']}",
                    "--set-param", f"model.extra_decision_tree.pipeline_type.frequency_encoding={params['frequency_encoding']}",
                    "--set-param", f"model.extra_decision_tree.pipeline_type.KFold_frequency_encoding={params['KFold_frequency_encoding']}",
                    "--set-param", f"model.extra_decision_tree.pipeline_type.tfidf_vectorizer_encoding={params['tfidf_vectorizer_encoding']}",
                    "--set-param", f"model.extra_decision_tree.pipeline_type.count_vectorizer_encoding={params['count_vectorizer_encoding']}",

                    ]
                  )
    
#===============================================================================================

def run_exp_Random_Forest():
    
    #Hyperparamter tunn=ing for Random Forest
    params = {
        "n_estimators": random.choice([50, 100, 150, 200]),
        "max_depth": random.choice([4, 5]),

        "bootstrap": True,
        "max_samples": random.choice([0.5, 0.6, 0.7, 0.8, 0.95]),
        
        "max_features":  random.choice([0.2, 0.3, 0.5, 0.6, 0.95]),
        "min_samples_leaf":  random.choice([0.01, 0.05, 0.001, 0.002, 0.005]),

        "class_weight": 'balanced', #random.choice(['balanced', 'balanced_subsample']),

        #Select the featurization techinique
        "KFoldTE": random.choice([True, False]), 
        "frequency_encoding": random.choice([True, False]),
        "KFold_frequency_encoding": random.choice([True, False]),
        "tfidf_vectorizer_encoding": random.choice([True, False]),
        "count_vectorizer_encoding": random.choice([True, False]),
    }

    subprocess.run(["dvc", "exp", "run", 
                    "--set-param", f"model.random_forest.hyper_params.n_estimators={params['n_estimators']}",
                    "--set-param", f"model.random_forest.hyper_params.max_depth={params['max_depth']}",

                    "--set-param", f"model.random_forest.hyper_params.bootstrap={params['bootstrap']}",
                    "--set-param", f"model.random_forest.hyper_params.max_samples={params['max_samples']}",

                    "--set-param", f"model.random_forest.hyper_params.max_features={params['max_features']}",
                    "--set-param", f"model.random_forest.hyper_params.min_samples_leaf={params['min_samples_leaf']}",

                    "--set-param", f"model.random_forest.hyper_params.class_weight={params['class_weight']}",


                    #Select the featurization techinique
                    "--set-param", f"model.random_forest.pipeline_type.KFoldTE={params['KFoldTE']}",
                    "--set-param", f"model.random_forest.pipeline_type.frequency_encoding={params['frequency_encoding']}",
                    "--set-param", f"model.random_forest.pipeline_type.KFold_frequency_encoding={params['KFold_frequency_encoding']}",
                    "--set-param", f"model.random_forest.pipeline_type.tfidf_vectorizer_encoding={params['tfidf_vectorizer_encoding']}",
                    "--set-param", f"model.random_forest.pipeline_type.count_vectorizer_encoding={params['count_vectorizer_encoding']}",

                    ]
                  )

#===============================================================================================

def run_exp_XBoost():
    
    #Hyperparamter tunn=ing for Random Forest
    params = {
        "n_estimators": random.choice(range(50, 500, 5)), #random.choice([50, 100, 150, 200]),
        "max_depth": random.choice([4, 5]),
        "reg_lambda": random.choice(np.arange(1, 2, 0.01)), 
        "learning_rate": random.choice(np.arange(0.1, 1, 0.01)),
        "colsample_bytree": random.choice(np.arange(0.6, 1, 0.005)),
        "random_state": random.choice(range(50, 500, 30)),
    }

    subprocess.run(["dvc", "exp", "run", 
                    "--set-param", f"model.xgboost.hyper_params.n_estimators={params['n_estimators']}",
                    "--set-param", f"model.xgboost.hyper_params.max_depth={params['max_depth']}",                   

                    "--set-param", f"model.xgboost.hyper_params.reg_lambda={params['reg_lambda']}",
                    "--set-param", f"model.xgboost.hyper_params.learning_rate={params['learning_rate']}",
                    "--set-param", f"model.xgboost.hyper_params.colsample_bytree={params['colsample_bytree']}",                 

                    ]
                  )

#===============================================================================================



def run_exp_Logistic_Reg():
    
    #Hyper paramater tuning for Logistic Regression 
    params = {
        "max_iter": random.choice(range(5,100,5)),#random.choice([500, 600, 700, 800, 900, 1000]),
        "penalty": random.choice(['l2']), #'l1': 
        "C": random.choice(range(1,100,1)), #random.choice([10**-4, 10**-2, 10**0, 10**1, 10**4]),

        #Select the featurization techinique
        "KFoldTE": True, #random.choice([True, False]), 
        "frequency_encoding": False, #random.choice([True, False]),
        "KFold_frequency_encoding": False, #random.choice([True, False]),
        "tfidf_vectorizer_encoding": False, #random.choice([True, False]),
        "count_vectorizer_encoding": True, #random.choice([True, False]),
    }

    #This will generate the experiment and wait for instruction to execute 
    #--temp : did not help. Continue to run from ".dvc\tmp\exps\"
    # Initial thought to use --queue, since they will run ".dvc\tmp\exps\" environment it alwasy gave error when finding the data file. This is usefull when using remort storage systems
    subprocess.run(["dvc", "exp", "run", #"--queue",
                    "--set-param", f"model.logistic_reg.hyper_params.max_iter={params['max_iter']}",
                    "--set-param", f"model.logistic_reg.hyper_params.penalty={params['penalty']}",
                    "--set-param", f"model.logistic_reg.hyper_params.C={params['C']}",

                    #Select the featurization techinique
                    "--set-param", f"model.logistic_reg.pipeline_type.KFoldTE={params['KFoldTE']}",
                    "--set-param", f"model.logistic_reg.pipeline_type.frequency_encoding={params['frequency_encoding']}",
                    "--set-param", f"model.logistic_reg.pipeline_type.KFold_frequency_encoding={params['KFold_frequency_encoding']}",
                    "--set-param", f"model.logistic_reg.pipeline_type.tfidf_vectorizer_encoding={params['tfidf_vectorizer_encoding']}",
                    "--set-param", f"model.logistic_reg.pipeline_type.count_vectorizer_encoding={params['count_vectorizer_encoding']}",

                    ]
                  #This did not help
                  #, cwd = get_project_root()
                  #, cwd = wrk_dir 
                  )
    
for _ in tqdm (range(num_exps), desc = "Generating dvc exp..."):

    #run_exp_Logistic_Reg()
    
    #run_exp_Random_Forest()

    #run_exp_ExtraTreesClassifier()
    
    run_exp_Decision_Tree()
    
    #run_exp_XBoost()

print("Queued Experiement to run.")
#=============================================================

#Step 1: Run the file to generate the exp queue
#       python src\data\random_search.py
#Step 2: Either you can selected the exp from the dvc studio to run or execute cmd to run all the queue exp
#       dvc exp run --run-all #This will cause to run the expirement from ".dvc\tmp\exps\" dir of the current workspace
# To run within environment use: dvc queue start
"""
#Example
# Automated grid search experiments
max_iter = [250, 300, 350, 400, 450, 500]
penalty = [8, 16, 32, 64, 128, 256]

# Iterate over all combinations of hyperparameter values.
for n_est, min_split in itertools.product(n_est_values, min_split_values):
    # Execute "dvc exp run --queue --set-param train.n_est=<n_est> --set-param train.min_split=<min_split>".
    subprocess.run(["dvc", "exp", "run", "--queue",
                    "--set-param", f"train.n_est={n_est}",
                    "--set-param", f"train.min_split={min_split}"])

"""