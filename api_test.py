import requests
import json

url = "http://127.0.0.1:4000/employee_resource_access/check_access_to_resource"

resource_detail = json.dumps({
  "resource": 78766,
  "role_rollup_1": 118079,
  "role_rollup_2": 118080,
  "role_family": 19721,
  "role_family_desc": 118177,
  "role_deptname": 117878,
  "role_code": 117880
})

headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers = headers, data = resource_detail)

print(response.text)