# Employee Access Resource #

When an employee at any company starts work, they first need to obtain the computer access necessary to fulfill their role. This access may allow an employee to read/manipulate resources through various applications or web portals.  

## Objective ##

The objective is to build a model, learned using historical data, that will determine an employee's access needs. So model will take an employee's role information and a resource code and will return whether or not access should be granted. This is a binary classification task.

## Dataset ##

### Source: https://www.kaggle.com/competitions/amazon-employee-access-challenge

### Description of the features: ###
Given the data related to current employees and their provisioned access

* Label : ACTION is 1 if access is granted to the resource and 0 if not granted.
* Features
    - RESOURCE - An ID for each resource
    - MGR_ID - The EMPLOYEE ID of the manager of the current EMPLOYEE ID record. An employee may have only one manager at a time.
    - ROLE_ROLLUP_1 - Company role grouping category id 1 (e.g. US Engineering)
    - ROLE_ROLLUP_2 - Company role grouping category id 2 (e.g. US Retail)
    - ROLE_DEPTNAME - Company role department description (e.g. Retail)
    - ROLE_TITLE - Company role business title description (e.g. Senior Engineering Retail Manager)
    - ROLE_FAMILY_DESC - Company role family extended description (e.g. Retail Manager, Software Engineering)
    - ROLE_FAMILY - Company role family description (e.g. Retail Manager)
    - ROLE_CODE - Company role code; this code is unique to each role (e.g. Manager)

We have large class imbalance.

# Metric #

Resource access may allow an employee to read/manipulate resources through various applications or web portals. So its crutual that employee does not get  access to Resource they should not have. 
Recall is the score that will make sure the False negartive.

Therefore F1-score is a good metric to evaluate the performance of this dataset as it weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.



# Solution # 


Dataset in imbalance. 
 
Feature enginnner by combininti varaious feature and calclaueing TFIDFVectorization and CountVectoris to generated feature whocj will be combination of the features


# Experiments Results #


# How to setup the local environment #

Download and install DVC version dvc-2.34.0

run comand 
to init dvc in the project. GO to project ternminal and type : dvc init
When we run dvc repro there are chnages in dvc.lock and we need to manully call git add dvc.lock to commit stage
or excute command after dvc init. 
"dvc config core.autostage true" :  if enabled, DVC will automatically stage (git add) DVC files created or modified by DVC commands. 


Сreate and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt


To reproduce any stage
dvc repro
dvc repro --force: to force all the stage
Run the experiant with the value in param: dvc exp run
On the fly chnage value of param: dvc exp run -S src\data\train_params.yaml:model.params.C=1.5  -S  src\data\train_params.yaml:model.params.max_iter=1000
Note: No need to specify "src\data\train_params.yaml" if the param.yaml is the default param. this is used in case of custom param 



Installation
Python 3.8+ is required to run code from this repo.

$ git clone https://github.com/iterative/demo-bank-customer-churn
$ cd demo-bank-customer-churn
Now let's install the requirements. But before we do that, we strongly recommend creating a virtual environment with a tool such as virtualenv:

$ virtualenv -p python3 .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
Running in your environment
Create and configure a location for remote storage (e.g. AWS S3) OR setup a local DVC remote. Change the pointer to your DVC remote in .dvc/config.
Download Churn_Modeling.csv file from here and place it in data/Churn_Modelling.csv
Now you can start a Jupyter Notebook server and execute the notebook notebook/TrainChurnModel.ipynb top to bottom to train a model

$ jupyter notebook
If you want to run the DVC pipeline:

dvc repro # runs the pipeline defined in `dvc.yaml`
dvc push # pushes the resulting artifacts to a DVC remote configured in `.dvc/config`


https://github.com/Machine-Learning-01/sensor-fault-detection
