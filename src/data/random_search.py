from utils.paths import get_project_root
from tqdm import tqdm
import subprocess
import itertools
import random
import os


# Automated random search experiments
num_exps = 50 #Number of experiments to run to generate
random.seed(0)


"""
i = 3 #Since in my project structure are max at 3 level
wrk_dir = os.getcwd()
while i > 0:  
    if not os.path.exists(os.path.join(wrk_dir, 'params.yaml')):       
        wrk_dir = os.path.abspath(os.path.join(wrk_dir, os.pardir))        
    i = i - 1

print(f'Working directory: {wrk_dir}')
"""

for _ in tqdm (range(num_exps), desc = "Generating dvc exp..."):

    #Hyper-paramters for LogRegression
    params = {
        "max_iter": random.choice([50, 100, 120, 130, 150, 200, 250]),
        "penalty": random.choice(['l2']), #'l1': 
        "C":  random.choice([100, 10, 1.0, 0.1, 0.01]),

        #Select the featurization techinique
        "KFoldTE": random.choice([True, False]), 
        "frequency_encoding": random.choice([True, False]),
        "KFold_frequency_encoding": random.choice([True, False]),
        "tfidf_vectorizer_encoding": random.choice([True, False]),
        "count_vectorizer_encoding": random.choice([True, False]),
    }

    #This will generate the experiment and wait for instruction to execute 
    #--temp : did not help. Continue to run from ".dvc\tmp\exps\"
    # Initial thought to use --queue, since they will run ".dvc\tmp\exps\" environment it alwasy gave error when finding the data file. This is usefull when using remort storage systems
    subprocess.run(["dvc", "exp", "run", #"--queue",
                    "--set-param", f"model.logistic_reg.hyper_params.max_iter={params['max_iter']}",
                    "--set-param", f"model.logistic_reg.hyper_params.penalty={params['penalty']}",
                    "--set-param", f"model.logistic_reg.hyper_params.C={params['C']}",

                    #Select the featurization techinique
                    "--set-param", f"pipeline_type.KFoldTE={params['KFoldTE']}",
                    "--set-param", f"pipeline_type.frequency_encoding={params['frequency_encoding']}",
                    "--set-param", f"pipeline_type.KFold_frequency_encoding={params['KFold_frequency_encoding']}",
                    "--set-param", f"pipeline_type.tfidf_vectorizer_encoding={params['tfidf_vectorizer_encoding']}",
                    "--set-param", f"pipeline_type.count_vectorizer_encoding={params['count_vectorizer_encoding']}",

                    ]
                  #This did not help
                  #, cwd = get_project_root()
                  #, cwd = wrk_dir 
                  )

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