from src.models.predict_model import employee_access_resource
from utils.read_utils import read_yaml_key
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn 


model_path = read_yaml_key('trained_model.model_path') #'./model/model.pkl'
feature_pipeline = read_yaml_key('trained_model.feature_eng') #'./model/feature_engg_pipeline.json'
model_obj = employee_access_resource(model_path, 
                                     feature_pipeline
                                     )


app = FastAPI(title = "Employee Resource Access") #api code using FASTAPI

def is_digit(value):
    return str.isdigit(value)

class ResourceDetails(BaseModel):
    resource: int
    role_rollup_1: int
    role_rollup_2: int
    role_family: int
    role_family_desc: int
    role_deptname: int
    role_code: int

@app.get("/employee_resource_access")
def index():
    return 'API is working as expected. Now head over to "/employee_resource_access/check_access_to_resource" for employee access check.'
    #return render_template('index.html')


@app.post("/employee_resource_access/check_access_to_resource")
def predict(data: ResourceDetails):
    
    data = data.dict()

    """
    #Not need after defining ResourceDetails calss using pydantic 
    
    if not is_digit(data['resource']):
        raise HTTPException(status_code = 415, detail = "Select valid Resource from the drop down.")

    elif not is_digit(data['role_rollup_1']):
        raise HTTPException(status_code = 415, detail = "Select valid Role_RollUp_1 from the drop down.")        
    elif not is_digit(data['role_rollup_2']):
        raise HTTPException(status_code = 415, detail = "Select valid Role_RollUp_2 from the drop down.")

    elif not is_digit(data['role_family']):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Family from the drop down.")  
    elif not is_digit(data['role_family_desc']):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Family_Desc from the drop down.")  

    elif not is_digit(data['role_deptname']):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Deptname from the drop down.")  
    elif not is_digit(data['role_code']):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Code from the drop down.") 
    """

    score, result = model_obj.check_access(data['resource'], 
                                           data['role_rollup_1'], data['role_rollup_2'], 
                                           data['role_family'], data['role_family_desc'], 
                                           data['role_deptname'], data['role_code']
                                    )

    #print(result)
    return {'prediction_score': score, 'prediction': result}

if __name__ == '__main__':

    uvicorn.run(app, host = '127.0.0.1', port = 4000, debug = True)




