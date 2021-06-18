# service_builder
1. Run the following command
 
     ``` 
     docker  build  -t dpm360_model_wrapper_v1  .
    
     ```

2. Tag the image and push to registry
    NB: I am using my public git account: kibnelson
    
     ``` 
     docker  tag dpm360_model_wrapper_v1 kibnelson/dpm360_model_wrapper_v1
    
     ```
3. Push the image   
  
     ``` 
     docker  push kibnelson/dpm360_model_wrapper_v1
    
     ```
 # Testing    
1. To test image locally run the following command
 
    ```
    docker run --publish=0.0.0.0:8080:8080 -e MLFLOW_S3_ENDPOINT_URL=https://foh-ohdsi-dev-minio.bx.cloud9.ibm.com -e AWS_ACCESS_KEY_ID=minioRoot -e AWS_SECRET_ACCESS_KEY=minioRoot123 -e MODEL_NAME=ElasticnetWineTestModel -e MODEL_VERSION=1 -e MODEL_RUN_ID=208797568cee4b1c96c121827eb1ddac -e MLFLOW_TRACKING_URI=https://foh-ohdsi-dev-mlflow.bx.cloud9.ibm.com -e MODEL_SOURCE=s3://mlflow-experiments/0/208797568cee4b1c96c121827eb1ddac/artifacts/dpm360 -ti dpm360_model_wrapper_v2

    ``` 
2. To test model wrapper code locally, set the following env variables 
     ``` 
    export MODEL_NAME=ElasticnetWineModelFeatures3
    export MODEL_VERSION=1
    export FEATURE_GENERATOR_FILE_NAME=""
    export MLFLOW_TRACKING_URI=https://foh-ohdsi-dev-mlflow.bx.cloud9.ibm.com
    export MODEL_SOURCE=s3://mlflow-experiments/0/81e4192736f8497384e09d6928ee0f2f/artifacts/model
    export MODEL_RUN_ID=81e4192736f8497384e09d6928ee0f2f
    export AWS_ACCESS_KEY_ID=minioRoot
    export AWS_SECRET_ACCESS_KEY=minioRoot123
    export MLFLOW_S3_ENDPOINT_URL=https://foh-ohdsi-dev-minio.bx.cloud9.ibm.com
    
     ```
3. Then start flask app
    ```
    python ModelWrapperApp.py    
    
    ```
  
    To access the swagger page
    
    ```
    http://0.0.0.0:8080/doc/
    
    ```       
    The following payload is used to test the sample elasticwine model 
    
    ```
    
    {"alcohol":[12.8],"chlorides":[0.029],"citricacid":[0.029],"density":[0.029],"fixedacidity":[0.029],"freesulfurdioxide":[0.029],"pH":[0.029],"residualsugar":[0.029],"sulphates":[0.029],"totalsulfurdioxide":[0.029],"volatileacidity":[0.029]}
    
    ```
# Access the deployed pod
1. Run the following to login to the cluster
   
   ```
   oc login --token=ogRUKtHV5kmece-W8cyKW6WNOKtP520BaIQ5FKJTp_A --server=https://c100-e.us-south.containers.cloud.ibm.com:30049
   ```
2. Run the following command to see new pods 
   ```
    oc get pods
   
    # or use kubectl
   kubectl get pods    
   ```   
 2. Get the name of the pod of interest
   ```
       oc port-forward elasticnetwinemodel-65b94cbc44-tlp8t 8080:8080 
   ```  
   Thereafter access the following address to access the pods swagger
   ```
    http://0.0.0.0:8080/doc/

   ```
   The following payload is used to test the sample elasticwine model 
    
    ```
    
    {"alcohol":[12.8],"chlorides":[0.029],"citricacid":[0.029],"density":[0.029],"fixedacidity":[0.029],"freesulfurdioxide":[0.029],"pH":[0.029],"residualsugar":[0.029],"sulphates":[0.029],"totalsulfurdioxide":[0.029],"volatileacidity":[0.029]}
    
    ```

# Local cluster set up
Run the following commands inside service_builder dir
1. To start, we pass start command. After the integration with training pipeline we will also pass model_name to  be used to build  an image
```
sh startup.sh start 

```

2. To stop
```
sh startup.sh stop

```

NB: Once we have the training pipeline ready we will follow the example below to integrate

```
https://docs.seldon.io/projects/seldon-core/en/v0.3.0/examples/mlflow.html
```

# Troubleshooting

If the deployment does not work, possible todo following this https://www.kubeflow.org/docs/started/workstation/minikube-linux/

# Helm NOTES

Use the following command to helm chart outputs

```
helm template --dry-run --debug service_builder ./chart --set image.name=slur

```
To install 

```

helm install service_builder ./chart --set image.name=slur

```


Sample patient test id

```
subject_id = [392786844,392814153,392798611])

```
