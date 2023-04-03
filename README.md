# Employee Access Resource #

When an employee at any company starts work, they first need to obtain the computer access necessary to fulfill their role. This access may allow an employee to read/manipulate resources through various applications or web portals.  

## Objective ##

The objective is to build a model, learned using historical data, that will determine an employee's access needs. So model will take an employee's role information and a resource code and will return whether or not access should be granted. 
<br>
This is a binary classification task.
<br>
Resource access may allow an employee to read/manipulate resources through various applications or web portals. So its crucial that employee does not get access to Resource they should not have. Recall would be good metric to evaluate false negative. Therefore F1-score is a good metric to evaluate the performance of this dataset as it weights recall and precision equally, and a good retrieval algorithm will maximize both precision and recall simultaneously.

## Dataset ##

### Source: https://www.kaggle.com/competitions/amazon-employee-access-challenge

### Description of the features: ###
Given the data related to current employees and their provisioned access

* We have large class imbalance.
* Features
    - ACTION - Class label to grant access to resource. If 1 access is granted to the resource else no access.
    - RESOURCE - An ID for each resource
    - MGR_ID - The EMPLOYEE ID of the manager of the current EMPLOYEE ID record. An employee may have only one manager at a time.
    - ROLE_ROLLUP_1 - Company role grouping category id 1 (e.g. US Engineering)
    - ROLE_ROLLUP_2 - Company role grouping category id 2 (e.g. US Retail)
    - ROLE_DEPTNAME - Company role department description (e.g. Retail)
    - ROLE_TITLE - Company role business title description (e.g. Senior Engineering Retail Manager)
    - ROLE_FAMILY_DESC - Company role family extended description (e.g. Retail Manager, Software Engineering)
    - ROLE_FAMILY - Company role family description (e.g. Retail Manager)
    - ROLE_CODE - Company role code; this code is unique to each role (e.g. Manager)

# How to setup the local environment #

### Step 1: Installation ###

Required to run code from this repo.

        Python 3.9+ 
        dvc-2.34.0

### Step 2:  Clone the repository ###

        $ git clone https://github.com/SmithaUpadhyaya/employee_access_resource_classifier.git
        $ cd employee_access_resource_classifier

### Step 3: Сreate and activate a virtual environment ###

Creating a virtual environment and install all the requirements

    python -m venv .env
    source .env/bin/activate
    pip install --upgrade pip
    
### Step 4: Install the requirements ###    

    pip install -r requirements.txt

### Step 5: Configure DVC remotre storage ###

    Create and configure a location for remote storage (e.g. AWS S3) OR setup a local DVC remote. Change the pointer to your DVC remote in .dvc/config.

### Step 6: Download the data ###

    Download and extract zip "amazon-employee-access-challenge.zip" and place the data files in "data/training" folder

### Step 7: Run the application ###
<br>

#### -> Run from local environment ####
<br>
    1. Open terminal and execute. 

        uvicorn app:app --reload

    2. Open any web browser and type in URL:  

        http://127.0.0.1:8000/employee_resource_access

<br>

#### -> Run from Docker image ####

    1. Build the Docker image

        docker build -t employee_access_check .

    2. Run the Docker image

        docker run -d --name cp_employee_access_check -p 80:80 employee_access_check

    3. Open any web browser and type in URL:

        http://127.0.0.1:8000/employee_resource_access

<br>

# DVC pipeline #

 * Reproduce DVC pipeline 
    
        dvc repro # runs the pipeline defined in 'dvc.yaml' and param defined in 'param.yaml'
        dvc push # pushes the resulting artifacts to a DVC remote configured in `.dvc/config`

 * Run experiments
 
   Change paramaters in 'param.yaml' file then run
    
        dvc exp run

 * Run grid search on paramaters
    
    Change/Define the paramaters for the model in 'src\data\random_search.py'

        python src\data\random_search.py

# **Tech Stack Used** #
</br>
1) Python 3.9.8 </br>
2) Random Forest - Ensemble Technique</br>
3) DVC - Pipeline and Track experiments </br>
4) FastAPI </br>
5) Docker </br>
</br>

# **Demo** #
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="references\demo.gif" />


