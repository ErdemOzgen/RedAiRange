import requests
from urllib.parse import quote

pipeline = "AI-Validate New Model"
encoded = quote(pipeline)

jenkins_url = f"http://localhost:8080/job/AI-Validate%20New%20Mode/buildWithParameters"
params = {'token': 'AISEC', 'MODEL_FILE_PATH': 'simple-cifar10.h5'}
auth = ('jsotiro', '<your account token>')

response = requests.post(jenkins_url, params=params,auth=auth )

if response.status_code == 201:
    print("Successfully triggered the ValidateNewModel pipeline.")
else:
    print("Failed to trigger the pipeline. Status code:", response.status_code)
