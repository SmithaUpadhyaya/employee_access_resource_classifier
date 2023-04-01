import uvicorn #We define out API using fastapi but serving is done using uvicorn
from src.models.predict_model import employee_access_resource
from utils.read_utils import read_yaml_key
from fastapi import FastAPI, HTTPException#, render_template
from pydantic import BaseModel

model_path = read_yaml_key('trained_model.model_path') #'./model/model.pkl'
feature_pipeline = read_yaml_key('trained_model.feature_eng') #'./model/feature_engg_pipeline.json'
model_obj = employee_access_resource(model_path, 
                                     feature_pipeline
                                     )


app = FastAPI(title = "Employee Resource Access") #api code using FASTAPI

def is_digit(value):
    return str.isdigit(value)

class ResourceDetails(BaseModel):
    f_name: str
    l_name: str
    phone_number: int

@app.get("/employee_resource_access/")
def home():
    return 'API is working as expected. Now head over to "/employee_resource_access/check_access_to_resource" for employee access check.'
    #return render_template('index.html')

#We can host multiple on the same server, by assign a different endpoint to each model. Best practices is to include the name model usercase in the endpoint.
#For API request, used uery-based approach where the parameters are passed by appending the “?” at the end of the URL and using “&” to add multiple parameters 
@app.post("/employee_resource_access/check_access_to_resource")
def predict(resource: str, 
            role_rollup_1: str, role_rollup_2: str, 
            role_family: str, role_family_desc: str, 
            role_deptname: str, role_code: str
            ):

    if not is_digit(resource):
        raise HTTPException(status_code = 415, detail = "Select valid Resource from the drop down.")

    elif not is_digit(role_rollup_1):
        raise HTTPException(status_code = 415, detail = "Select valid Role_RollUp_1 from the drop down.")        
    elif not is_digit(role_rollup_2):
        raise HTTPException(status_code = 415, detail = "Select valid Role_RollUp_2 from the drop down.")

    elif not is_digit(role_family):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Family from the drop down.")  
    elif not is_digit(role_family_desc):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Family_Desc from the drop down.")  

    elif not is_digit(role_deptname):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Deptname from the drop down.")  
    elif not is_digit(role_code):
        raise HTTPException(status_code = 415, detail = "Select valid Role_Code from the drop down.") 

    result = model_obj.check_access(resource, 
                                    role_rollup_1, role_rollup_2, 
                                    role_family, role_family_desc, 
                                    role_deptname, role_code
                                    )

    #print(result)
    return result

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)




