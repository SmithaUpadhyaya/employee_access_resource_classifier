# Employee Access Resource
When an employee at any company starts work, they first need to obtain the computer access necessary to fulfill their role. This access may allow an employee to read/manipulate resources through various applications or web portals.The objective is to build a model, learned using historical data, that will determine an employee's access needs.


Download and install DVC version dvc-2.34.0

run comand 
to init dvc in the project. GO to project ternminal and type : dvc init
When we run dvc repro there are chnages in dvc.lock and we need to manully call git add dvc.lock to commit stage
or excute command after dvc init. 
"dvc config core.autostage true" :  if enabled, DVC will automatically stage (git add) DVC files created or modified by DVC commands. 


# How to setup the local environment #
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

